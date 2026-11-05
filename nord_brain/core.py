"""Small *own-weights* decoder-only Transformer inspired by Qwen's building blocks.

RMSNorm, causal grouped-query attention, RoPE, and SwiGLU. This is NOT Qwen,
contains NO pretrained weights, and is not expected to speak fluently at birth.

The model learns via genuine PyTorch autograd and can grow its MLP channels
while preserving its existing forward output at the instant of expansion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

VOCAB_SIZE = 258  # UTF-8 byte IDs [0,255], EOS 256, BOS 257
EOS, BOS = 256, 257


@dataclass
class TransformerConfig:
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    kv_heads: int = 2
    intermediate_size: int = 384
    context: int = 256

    def validate(self):
        if not 32 <= self.d_model <= 512 or self.d_model % self.heads:
            raise ValueError('Dimensão do modelo incompatível com heads')
        if not 1 <= self.layers <= 12 or not 1 <= self.heads <= 16:
            raise ValueError('Número de camadas ou cabeças fora dos limites')
        if self.heads % self.kv_heads or self.kv_heads < 1:
            raise ValueError('GQA precisa de heads divisíveis por kv_heads')
        if self.d_model // self.heads % 2:
            raise ValueError('RoPE exige head_dim par')
        if not self.d_model <= self.intermediate_size <= 4096:
            raise ValueError('Dimensão intermediária fora dos limites')
        if not 64 <= self.context <= 2048:
            raise ValueError('Janela de contexto fora dos limites')


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def rotary_embedding(x: torch.Tensor) -> torch.Tensor:
    # x: [batch, heads, sequence, head_dim]
    t, d = x.shape[-2:]
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    angles = torch.outer(torch.arange(t, device=x.device, dtype=torch.float32), inv)
    sin = angles.sin()[None, None, :, :]
    cos = angles.cos()[None, None, :, :]
    left, right = x.float()[..., : d // 2], x.float()[..., d // 2 :]
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1).to(x.dtype)


class GroupedAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.d_model // config.heads
        self.q_proj = nn.Linear(config.d_model, config.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        def reshape(y, heads):
            return y.view(b, t, heads, self.head_dim).transpose(1, 2)
        q = rotary_embedding(reshape(self.q_proj(x), self.heads))
        k = rotary_embedding(reshape(self.k_proj(x), self.kv_heads))
        v = reshape(self.v_proj(x), self.kv_heads)
        if self.kv_heads != self.heads:
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0)
        return self.o_proj(a.transpose(1, 2).contiguous().reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, width: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, width, bias=False)
        self.up_proj = nn.Linear(d_model, width, bias=False)
        self.down_proj = nn.Linear(width, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def grow(self, additional: int):
        """Widen MLP preserving output exactly at expansion time (zero down columns)."""
        old = self.gate_proj.out_features
        device = self.gate_proj.weight.device
        dtype = self.gate_proj.weight.dtype
        d = self.gate_proj.in_features
        new = old + additional
        gate = nn.Linear(d, new, bias=False, device=device, dtype=dtype)
        up = nn.Linear(d, new, bias=False, device=device, dtype=dtype)
        down = nn.Linear(new, d, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            gate.weight[:old].copy_(self.gate_proj.weight)
            up.weight[:old].copy_(self.up_proj.weight)
            down.weight[:, :old].copy_(self.down_proj.weight)
            down.weight[:, old:].zero_()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GroupedAttention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class NordTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embed = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, VOCAB_SIZE, bias=False)
        # Tied embeddings: smaller total parameter and consistent input/output semantics.
        self.lm_head.weight = self.embed.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2 or tokens.size(1) > self.config.context:
            raise ValueError('Entrada excede janela de contexto')
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def grow_mlp(self, additional: int):
        if not 8 <= additional <= 256 or self.config.intermediate_size + additional > 4096:
            raise ValueError('Expansão entre 8 e 256, máximo de 4096 canais MLP')
        for block in self.blocks:
            block.mlp.grow(additional)
        self.config.intermediate_size += additional


class NeuralCore:
    """Simple persistent online-training API. Stores model AND optimizer state."""
    def __init__(self, path: Path, config: TransformerConfig | None = None, seed: int = 17):
        self.path = Path(path)
        self.steps = 0
        # Avoid flooding a 1-vCPU machine with dozens of CPU worker threads.
        torch.set_num_threads(max(1, min(4, int(os.environ.get('NORD_THREADS', '1')))))
        if self.path.exists():
            state = torch.load(self.path, map_location='cpu', weights_only=True)
            if int(state.get('version', 0)) != 2:
                raise ValueError('Checkpoint não pertence ao Nord Transformer V2')
            self.model = NordTransformer(TransformerConfig(**state['config']))
            self.model.load_state_dict(state['model'])
            self.steps = int(state['steps'])
            self.optimizer = self._optimizer()
            self.optimizer.load_state_dict(state['optimizer'])
        else:
            torch.manual_seed(seed)
            self.model = NordTransformer(config or TransformerConfig())
            self.optimizer = self._optimizer()

    def _optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)

    @property
    def hidden_size(self):
        # Retained for compatibility with the old terminal command; this is MLP width.
        return self.model.config.intermediate_size

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix='.nord-transformer-', suffix='.pt', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                torch.save({'version': 2, 'config': asdict(self.model.config),
                            'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                            'steps': self.steps}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def load(self):
        """Recover last durable checkpoint after an interrupted/failed lesson."""
        restored = NeuralCore(self.path)
        self.model, self.optimizer, self.steps = restored.model, restored.optimizer, restored.steps

    @staticmethod
    def _example(prompt: str, answer: str, context: int):
        # Only target the answer's tokens, not the question text.
        prefix = ((prompt + '\nNord: ') if dialogue else
                  ('Pergunta: ' + prompt + '\nResposta: ')).encode('utf-8')
        suffix = answer.encode('utf-8')
        if len(prefix) + len(suffix) + 2 > context:
            raise ValueError(f'Exemplo demasiado longo para contexto de {context} tokens')
        full = [BOS] + list(prefix) + list(suffix) + [EOS]
        ids = torch.tensor(full[:-1], dtype=torch.long).unsqueeze(0)
        labels = torch.tensor(full[1:], dtype=torch.long).unsqueeze(0)
        labels[:, :len(prefix)] = -100  # masked prefix; first answer char at index len(prefix)
        return ids, labels

    def train(self, prompt: str, answer: str, epochs: int = 2, rate: float = 3e-4, *, dialogue: bool = False) -> float:
        if not 1 <= epochs <= 25 or not 0 < rate <= .01:
            raise ValueError('Épocas ou taxa de aprendizagem inválidas')
        windows = self._training_windows(prompt, answer, self.model.config.context, dialogue=dialogue)
        self.model.train()
        for group in self.optimizer.param_groups:
            group['lr'] = rate
        loss_value = float('nan')
        for _ in range(epochs):
            for tokens, labels in windows:
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(tokens)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=-100)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Erro numérico no treino')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.steps += 1
                loss_value = float(loss.detach())
        self.model.eval()
        return loss_value

    @staticmethod
    def _text_windows(text: str, context: int):
        """Supervised next-byte language modelling on a HUMAN-APPROVED corpus.

        Unlike question/answer training, all source bytes are next-token
        targets. Windows overlap for context, but each target is trained on
        once in the complete iteration. No automatic web/self-output labels.
        """
        clean=text.replace('\x00','').strip()
        if not clean or len(clean.encode('utf-8'))>100000:
            raise ValueError('O texto tem de conter entre 1 e 100000 bytes')
        full=[BOS]+list(clean.encode('utf-8'))+[EOS]
        stride=max(12,context//3)
        windows=[]
        for first in range(1,len(full),stride):
            end=min(len(full),first+stride)
            left=max(0,end-context)
            left=min(left,first-1)
            tokens=torch.tensor(full[left:end-1],dtype=torch.long).unsqueeze(0)
            labels=torch.tensor(full[left+1:end],dtype=torch.long).unsqueeze(0)
            labels[0,:max(0,first-left-1)]=-100
            windows.append((tokens,labels))
        return windows

    def train_text(self, text: str, *, updates: int=6, rate: float=.00012) -> float:
        """Update own Transformer weights from explicitly approved Portuguese.

        Sparse, deterministic sampling across the whole document prevents the
        first paragraph from monopolising short CPU training sessions.
        Caller owns checkpoint validation and persistence/rollback.
        """
        if not 1<=updates<=32 or not 0<rate<=.001:
            raise ValueError('Treino de texto: 1–32 atualizações e taxa segura')
        windows=self._text_windows(text,self.model.config.context)
        count=min(updates,len(windows))
        if count<1:
            raise ValueError('Texto vazio')
        # Rotate through different positions across later runs, with no RNG
        # dependency or hidden global state.
        offset=self.steps%len(windows)
        positions=[(offset+(i*len(windows))//count)%len(windows) for i in range(count)]
        self.model.train()
        for group in self.optimizer.param_groups:
            group['lr']=rate
        losses=[]
        try:
            for pos in positions:
                tokens,labels=windows[pos]
                self.optimizer.zero_grad(set_to_none=True)
                logits=self.model(tokens)
                loss=F.cross_entropy(logits.reshape(-1,VOCAB_SIZE),
                                     labels.reshape(-1),ignore_index=-100)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Erro numérico no treino textual')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
                self.optimizer.step()
                self.steps+=1
                losses.append(float(loss.detach()))
        finally:
            self.model.eval()
        return sum(losses)/len(losses)

    @torch.inference_mode()
    def score(self, examples: list[tuple[str, str]]) -> float:
        """Mean answer-token cross-entropy on evaluation examples, never trains."""
        if not examples:
            raise ValueError('Não existem exemplos de avaliação')
        self.model.eval()
        losses=[]
        for question,answer in examples:
            for ids,labels in self._training_windows(question,answer,self.model.config.context):
                logits=self.model(ids)
                per=F.cross_entropy(logits.reshape(-1,VOCAB_SIZE),labels.reshape(-1),
                                    ignore_index=-100,reduction='sum')
                tokens=int((labels!=-100).sum())
                if tokens:losses.append((float(per),tokens))
        return sum(value for value,n in losses)/sum(n for value,n in losses)

    @staticmethod
    def _training_windows(prompt: str, answer: str, context: int, *, dialogue: bool = False):
        """Full-answer supervision using bounded overlapping context windows.

        Each predicted answer byte is scored once; prompt bytes are context,
        not training targets. Handles UTF-8 and long explanations without
        silently discarding the answer.
        """
        prefix = ((prompt + '\nNord: ') if dialogue else ('Pergunta: ' + prompt + '\nResposta: ')).encode('utf-8')
        targets = answer.encode('utf-8')
        if not targets:
            raise ValueError('Resposta vazia')
        # The BOS or earliest prompt bytes may fall outside later windows.
        full = [BOS] + list(prefix) + list(targets) + [EOS]
        first_target = 1 + len(prefix)
        stride = max(12, context // 3)
        result=[]
        for first in range(first_target, len(full), stride):
            end = min(len(full), first + stride)
            left = max(0, end - context)
            # Training inputs predict tokens at positions left+1 .. end-1.
            left=min(left,first-1)
            ids=torch.tensor(full[left:end-1],dtype=torch.long).unsqueeze(0)
            labels=torch.tensor(full[left+1:end],dtype=torch.long).unsqueeze(0)
            mask=max(0,first-left-1)
            labels[0,:mask]=-100
            result.append((ids,labels))
        return result

    @torch.inference_mode()
    def generate(self, prompt: str, max_bytes: int = 100, temperature: float = .7, *, dialogue: bool = False) -> str:
        prefix = ((prompt + '\nNord: ') if dialogue else ('Pergunta: ' + prompt + '\nResposta: ')).encode('utf-8')
        # Keep enough room for generation; drop oldest prompt bytes if needed.
        max_length = self.model.config.context
        tokens = [BOS] + list(prefix)[-(max_length - 1):]
        result = bytearray()
        self.model.eval()
        for _ in range(max(1, min(int(max_bytes), max_length - 1, 300))):
            input_ids = torch.tensor(tokens[-max_length:],dtype=torch.long).unsqueeze(0)
            logits = self.model(input_ids)[0,-1].clone()
            logits[BOS] = -float('inf')
            if temperature <= 0:
                next_token = int(logits.argmax())
            else:
                topk = min(32, VOCAB_SIZE)
                values, indices = torch.topk(logits / max(.1,temperature), topk)
                next_token = int(indices[torch.multinomial(F.softmax(values,dim=-1),1).item()])
            if next_token == EOS:
                break
            result.append(next_token)
            tokens.append(next_token)
            if dialogue and (result.endswith(b'\nTu:') or
                             result.endswith(b'\nNord:')):
                del result[result.rfind(b'\n'):]
                break
        return result.decode('utf-8',errors='ignore').strip()

    def grow(self, extra: int = 16):
        """Widen all MLPs; preserve initial output, then reset optimizer state.

        Changing the parameter shapes invalidates Adam moments. The post-growth
        model is still untrained for the new connections.
        """
        self.model.grow_mlp(extra)
        self.optimizer = self._optimizer()
        return self.hidden_size
