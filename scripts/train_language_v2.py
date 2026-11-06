"""Nord Brain V2: isolated subword-token language experiment, CPU-first.

Run from repository root. Does not modify the V1 checkpoint.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from nord_brain.linguistic_training import TRAIN, DIALOGUE_TRAIN, HOLDOUT
from nord_brain.conversation_corpus import TRAINING as NATURAL_TRAIN
from nord_brain.dialogue_format import training_prompt

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
PATTERN = re.compile(r"\s+|[^\W\d_]+|\d+|[^\w\s]", re.UNICODE)


class Tokenizer:
    """Frequent word/subword units, with lossless UTF-8 byte fallback.

    Frequent tokens are entire words and whitespace units. Unknown words
    decompose to UTF-8 bytes. This is a hybrid word/byte tokenizer, not BPE.
    """
    def __init__(self, words):
        self.vocab = SPECIAL + [f"<b{i}>" for i in range(256)] + words
        self.lookup = {w: i for i, w in enumerate(self.vocab)}
        self.word_start = len(SPECIAL) + 256

    @classmethod
    def fit(cls, corpus, limit=4096):
        from collections import Counter
        counts = Counter(PATTERN.findall(corpus))
        words = [w for w, _ in counts.most_common(limit - 260) if w not in SPECIAL]
        return cls(words)

    def encode(self, text, bos=False, eos=False):
        out = [1] if bos else []
        for piece in PATTERN.findall(text):
            if piece in self.lookup:
                out.append(self.lookup[piece])
            else:
                out.extend(4 + b for b in piece.encode("utf-8"))
        if eos:
            out.append(2)
        return out

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            if 4 <= i < 260:
                buf.append(i - 4)
            elif i >= 260 and i < len(self.vocab):
                buf.extend(self.vocab[i].encode("utf-8"))
        return buf.decode("utf-8", errors="replace")

    def to_dict(self):
        return {"words": self.vocab[260:]}


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False, is_causal=True,
                         attn_mask=torch.ones(h.size(1), h.size(1), device=h.device,
                                              dtype=torch.bool).triu(1))
        x = x + a
        return x + self.ff(self.norm2(x))


class Model(nn.Module):
    def __init__(self, vocab, dim=192, layers=6, heads=6, context=256):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        b, n = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(n, device=ids.device))[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def evaluate(model, tokenizer, examples):
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for question, answer in examples:
            prefix = tokenizer.encode((question if question.startswith("Tu: ") else "Tu: " + question) + "\nNord: ", bos=True)
            suffix = tokenizer.encode(answer, eos=True)
            seq = (prefix + suffix)[-model.context - 1:]
            labels = torch.tensor(seq[1:], dtype=torch.long)
            start = max(0, len(prefix) - (len(prefix) + len(suffix) - len(seq)) - 1)
            labels[:start] = -100
            if (labels != -100).sum() == 0:
                continue
            logits = model(torch.tensor(seq[:-1], dtype=torch.long)[None])
            loss = F.cross_entropy(logits[0], labels, ignore_index=-100, reduction="sum")
            total += float(loss)
            count += int((labels != -100).sum())
    return total / max(1, count)


@torch.inference_mode()
def generate(model, tokenizer, prompt, temperature=0, max_tokens=80):
    model.eval()
    tokens = tokenizer.encode("Tu: " + prompt + "\nNord: ", bos=True)
    output = []
    for _ in range(max_tokens):
        logits = model(torch.tensor(tokens[-model.context:], dtype=torch.long)[None])[0, -1]
        logits[:4] = -float("inf")
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            vals, ix = torch.topk(logits / temperature, min(40, logits.numel()))
            nxt = int(ix[torch.multinomial(F.softmax(vals, -1), 1)])
        if nxt == 2:
            break
        output.append(nxt)
        tokens.append(nxt)
        if "\nTu:" in tokenizer.decode(output):
            break
    return tokenizer.decode(output).split("\nTu:")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("training/dataset/train.txt"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--rate", type=float, default=0.0002)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--data-dir", type=Path, default=Path.home() / ".nord-brain" / "v2")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.steps <= 100000 or not 128 <= args.context <= 512:
        ap.error("Invalid steps or context")
    if not 512 <= args.vocab <= 8192 or not 0 < args.rate <= 0.001:
        ap.error("Invalid vocab or rate")
    torch.set_num_threads(max(1, min(4, int(os.getenv("NORD_THREADS", "2")))))
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    corpus = args.corpus.read_text(encoding="utf-8")
    if len(corpus.encode("utf-8")) < 1000000:
        ap.error("Corpus must be at least 1 MB")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.data_dir / "language-v2.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        tokenizer = Tokenizer(state["tokenizer"]["words"])
        config = state["config"]
        model = Model(len(tokenizer.vocab), **config)
        model.load_state_dict(state["model"])
        optim = torch.optim.AdamW(model.parameters(), lr=args.rate)
        optim.load_state_dict(state["optimizer"])
        step = state["step"]
        print("Resuming V2 at step", step, flush=True)
    else:
        tokenizer = Tokenizer.fit(corpus, args.vocab)
        config = {"dim": 192, "layers": 6, "heads": 6, "context": args.context}
        model = Model(len(tokenizer.vocab), **config)
        optim = torch.optim.AdamW(model.parameters(), lr=args.rate)
        step = 0
        print("Created V2:", len(tokenizer.vocab), "tokens;",
              sum(p.numel() for p in model.parameters()), "parameters", flush=True)
    dialogues = list(TRAIN) + list(NATURAL_TRAIN) + [
        (training_prompt(turns, args.context), answer) for turns, answer in DIALOGUE_TRAIN
    ]
    rng.shuffle(dialogues)
    dialogue_index = 0
    # Document-level evaluation needs a separate corpus; this evaluates held-out dialogues only.
    print("Holdout before:", round(evaluate(model, tokenizer, HOLDOUT), 4), flush=True)
    text_tokens = tokenizer.encode(corpus)
    if len(text_tokens) < args.context + 1:
        ap.error("Not enough tokens")
    for i in range(args.steps):
        # First 70% of each 100-step block: general language; remainder: dialogue.
        language_phase = i % 100 < 70
        if language_phase:
            start = rng.randrange(len(text_tokens) - args.context)
            seq = text_tokens[start:start + args.context + 1]
            ids = torch.tensor(seq[:-1], dtype=torch.long)[None]
            labels = torch.tensor(seq[1:], dtype=torch.long)
        else:
            if dialogue_index and dialogue_index % len(dialogues) == 0:
                rng.shuffle(dialogues)
            question, answer = dialogues[dialogue_index % len(dialogues)]
            dialogue_index += 1
            prefix = tokenizer.encode("Tu: " + question + "\nNord: ", bos=True)
            suffix = tokenizer.encode(answer, eos=True)
            seq = prefix + suffix
            if len(seq) > args.context + 1:
                seq = seq[-args.context - 1:]
            ids = torch.tensor(seq[:-1], dtype=torch.long)[None]
            labels = torch.tensor(seq[1:], dtype=torch.long)
            masked = max(0, len(prefix) - (len(prefix) + len(suffix) - len(seq)) - 1)
            labels[:masked] = -100
        model.train()
        optim.zero_grad(set_to_none=True)
        logits = model(ids)[0]
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        step += 1
        if (i + 1) % 50 == 0 or i + 1 == args.steps:
            state = {"model": model.state_dict(), "optimizer": optim.state_dict(),
                     "tokenizer": tokenizer.to_dict(), "config": config, "step": step}
            tmp = checkpoint.with_suffix(".tmp")
            torch.save(state, tmp)
            os.replace(tmp, checkpoint)
            print(json.dumps({"step": step, "phase": "text" if language_phase else "dialogue",
                              "loss": round(float(loss), 4),
                              "holdout_loss": round(evaluate(model, tokenizer, HOLDOUT), 4)}), flush=True)
    if args.preview:
        for q in ("Olá", "Como estás hoje?", "Podemos conversar sobre uma ideia nova?"):
            print("PREVIEW", repr(q), repr(generate(model, tokenizer, q)), flush=True)


if __name__ == "__main__":
    main()
