"""Nord Brain curriculum trainer: words -> sentences -> reasoning -> prediction.

CPU-only, own Transformer, independent checkpoint. No external model/API.
Stage promotion is based on held-out token loss, NOT on step counts alone.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F
from train_language_v2 import Model, Tokenizer
from train_language_v3 import sample

STAGES = ("words", "sentences", "reasoning", "prediction")
# Held-out tasks are fixed and never included in the training examples.
EVAL = {
    "words": [("mar", "mar"), ("janela", "janela"), ("caminho", "caminho"), ("árvore", "árvore")],
    "sentences": [("Continua: A menina abriu", " a janela."), ("Continua: O cão correu", " pelo jardim."),
                  ("Continua: A chuva caiu", " durante a noite.")],
    "reasoning": [("Quanto é 7 + 5?", "12"), ("Quanto é 9 - 4?", "5"),
                  ("Quanto é 3 + 8?", "11")],
    "prediction": [("Se 3 + 4 = 7, quanto é 4 + 3?", "7"),
                   ("Depois de segunda-feira vem", " terça-feira."),
                   ("Se 6 - 2 = 4, quanto é 6 - 3?", "3")],
}
THRESHOLDS = {"words": 2.0, "sentences": 2.3, "reasoning": 1.7, "prediction": 1.7}


def examples(stage, corpus, rng):
    if stage == "words":
        # Word-in-context language modeling, not isolated vocabulary memorization.
        lines = [s.strip() for s in corpus.splitlines() if len(s.strip()) >= 30]
        if not lines:
            lines = [corpus]
        for _ in range(2000):
            line = rng.choice(lines)
            yield "Texto: " + line[:500] + "\n"
    elif stage == "sentences":
        lines = [s.strip() for s in corpus.splitlines() if len(s.strip()) >= 45]
        if not lines:
            lines = [corpus[i:i + 400] for i in range(0, len(corpus), 400)]
        for _ in range(2000):
            yield "Texto: " + rng.choice(lines)[:500] + "\n"
    elif stage in ("reasoning", "prediction"):
        for _ in range(2000):
            a, b = rng.randrange(0, 20), rng.randrange(0, 20)
            if stage == "reasoning":
                yield f"Pergunta: Quanto é {a} + {b}?\nResposta: {a+b}\n"
            else:
                yield f"Pergunta: Se tenho {a} objetos e recebo {b}, quantos tenho?\nResposta: {a+b}\n"


def task_loss(model, tok, pairs):
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for question, answer in pairs:
            prefix = tok.encode("Pergunta: " + question + "\nResposta: ", bos=True)
            suffix = tok.encode(answer, eos=True)
            seq = (prefix + suffix)[-model.context - 1:]
            shift = len(prefix) + len(suffix) - len(seq)
            labels = torch.tensor(seq[1:], dtype=torch.long)
            labels[:max(0, len(prefix) - shift - 1)] = -100
            if not (labels != -100).any():
                continue
            logits = model(torch.tensor(seq[:-1], dtype=torch.long)[None])[0]
            total += float(F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum"))
            count += int((labels != -100).sum())
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("training/dataset/train.txt"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--rate", type=float, default=0.0002)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--stage", choices=STAGES + ("auto",), default="auto")
    ap.add_argument("--data-dir", type=Path, default=Path.home() / ".nord-brain" / "curriculum")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.steps <= 100000 or not 128 <= args.context <= 512:
        ap.error("Invalid steps/context")
    if args.dim % args.heads or not 96 <= args.dim <= 384 or not 2 <= args.layers <= 10:
        ap.error("Invalid model shape")
    if not 512 <= args.vocab <= 8192 or not 0 < args.rate <= .001:
        ap.error("Invalid vocabulary/learning rate")
    torch.set_num_threads(max(1, min(4, int(os.getenv("NORD_THREADS", "2")))))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    corpus = args.corpus.read_text(encoding="utf-8")
    if len(corpus.encode("utf-8")) < 100000:
        ap.error("At least 100 KB of licensed training text required")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    path = args.data_dir / "curriculum.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        tok = Tokenizer(state["tokenizer"]["words"])
        model = Model(len(tok.vocab), **state["config"])
        model.load_state_dict(state["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate)
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.rate
        stage_idx = int(state["stage_idx"])
        step = int(state["step"])
        best = float(state.get("best", 999))
        print("Resuming curriculum:", STAGES[stage_idx], "step", step, flush=True)
    else:
        tok = Tokenizer.fit(corpus, args.vocab)
        config = dict(dim=args.dim, heads=args.heads, layers=args.layers, context=args.context)
        model = Model(len(tok.vocab), **config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate)
        stage_idx = 0
        step = 0
        best = 999.
        print("New curriculum:", sum(p.numel() for p in model.parameters()), "parameters", flush=True)
    config = dict(dim=model.embed.embedding_dim, heads=len(model.blocks[0].attn.in_proj_bias) // (3 * model.embed.embedding_dim) if False else model.blocks[0].attn.num_heads,
                  layers=len(model.blocks), context=model.context)
    if args.stage != "auto":
        selected = STAGES.index(args.stage)
        if selected > stage_idx:
            ap.error("Cannot skip stages; complete the earlier stages first")
        stage_idx = selected
    def save():
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     tokenizer=tok.to_dict(), config=config, stage_idx=stage_idx,
                     step=step, best=best)
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
    def report():
        return {stage: round(task_loss(model, tok, EVAL[stage]), 4)
                for stage in STAGES[:stage_idx + 1]}
    print("Evaluation:", json.dumps(report(), ensure_ascii=False), flush=True)
    if not args.eval_only:
        # Corpus windows are sampled as contiguous tokens; preserve real language structure.
        corpus_tokens = tok.encode(corpus)
        if len(corpus_tokens) <= model.context + 1:
            ap.error("Corpus too small for context")
        for i in range(args.steps):
            current = STAGES[stage_idx]
            if current in ("words", "sentences") or i % 5 == 0:
                start = rng.randrange(len(corpus_tokens) - model.context - 1)
                seq = corpus_tokens[start:start + model.context + 1]
                labels = torch.tensor(seq[1:], dtype=torch.long)
            else:
                text = next(examples(current, corpus, rng))
                seq = tok.encode(text, bos=True, eos=True)[-model.context - 1:]
                labels = torch.tensor(seq[1:], dtype=torch.long)
            ids = torch.tensor(seq[:-1], dtype=torch.long)[None]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(ids)[0], labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            step += 1
            if (i + 1) % 50 == 0 or i + 1 == args.steps:
                metrics = report()
                score = metrics[current]
                best = min(best, score)
                print(json.dumps({"step": step, "stage": current,
                                  "train_loss": round(loss.detach().item(), 4),
                                  "evaluation": metrics}, ensure_ascii=False), flush=True)
                # Promotion only after a minimum of 200 updates in current stage.
                # Thresholds are provisional and should be calibrated with better held-out tests.
                if args.stage == "auto" and (step >= 200) and score <= THRESHOLDS[current] and stage_idx < len(STAGES)-1:
                    stage_idx += 1
                    best = 999.
                    print("Stage promoted to:", STAGES[stage_idx], flush=True)
                save()
    if args.preview or args.eval_only:
        for q in ("Olá", "Como estás hoje?", "Quanto é 4 + 6?"):
            print("PREVIEW", repr(q), repr(sample(model, tok, q)), flush=True)


if __name__ == "__main__":
    main()
