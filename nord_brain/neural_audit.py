"""Read-only diagnostics for every Transformer block, including gradients."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from .core import BOS, EOS, VOCAB_SIZE


def audit(brain) -> str:
    with brain.lock:
        model = brain.core.model
        was_training = model.training
        model.eval()
        prompt = 'Tu: Olá! Como estás?\\nNord: '.replace('\\n', '\n')
        answer = 'Olá! Estou bem. E tu?'
        raw = [BOS] + list((prompt + answer).encode('utf-8')) + [EOS]
        x = torch.tensor([raw[:-1]], dtype=torch.long)
        labels = torch.tensor([raw[1:]], dtype=torch.long)
        labels[:, :len(prompt.encode('utf-8'))] = -100
        lines = [f'Auditoria neuronal: {len(model.blocks)} camadas, {brain.core.steps} passos']
        try:
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                   labels.reshape(-1), ignore_index=-100)
            loss.backward()
            lines.append(f'Perda do exemplo de diagnóstico: {loss.item():.4f}')
            with torch.no_grad():
                hidden = model.embed(x)
                for i, block in enumerate(model.blocks, 1):
                    attention = block.attn.o_proj.weight
                    mlp = block.mlp.down_proj.weight
                    ga = attention.grad
                    gm = mlp.grad
                    before = hidden
                    hidden = block(hidden)
                    contribution = (hidden - before).float().norm() / before.float().norm().clamp_min(1e-8)
                    finite = all(torch.isfinite(p).all().item() for p in block.parameters())
                    lines.append(
                        f'Camada {i}: atenção_grad={ga.norm().item() if ga is not None else 0:.6g}; '
                        f'mlp_grad={gm.norm().item() if gm is not None else 0:.6g}; '
                        f'contribuição={contribution.item():.5f}; finitos={finite}')
            lines.append('Diagnóstico sem atualização dos pesos; gradiente nulo numa projeção '
                         'não implica ausência de aprendizagem nas restantes.')
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)
        return '\n'.join(lines)
