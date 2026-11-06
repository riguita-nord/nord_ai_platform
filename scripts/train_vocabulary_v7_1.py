"""Nord Brain V7.1: progressive copy -> selection -> language extraction.

Independent checkpoint from V6. All model outputs are autoregressively generated.
Evaluation examples are never used for gradient updates. No curriculum stage 2 unlock.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from train_language_v2 import Model, Tokenizer
from train_vocabulary_v2 import TRAIN, TEST

KINDS = ("categoria", "propriedade", "contexto")
CONFIG = {"dim": 192, "heads": 6, "layers": 6, "context": 256}
# Byte-only tokenization prevents word-token shortcuts and permits novel strings.
# The 256 UTF-8 byte tokens can represent any well-formed UTF-8 word.
ALPHABET = "abcdefghijklmnopqrstuvxz"
FORMATS = (
    "Ficha de {w}: categoria={a}; propriedade={b}; contexto={c}. Pergunta: {q}",
    "Informação: {w} | contexto: {c} | categoria: {a} | propriedade: {b}. {q}",
    "Dados sobre {w}: propriedade {b}, categoria {a}, contexto {c}. {q}",
)
QUESTIONS = (
    "Qual é a {k} de {w}? Resposta: ",
    "Indica a {k} associada a {w}. Resposta: ",
    "Sobre {w}, diz a {k}. Resposta: ",
)
HELD_FORMATS = (
    "Registo [{w}]: contexto={c}; propriedade={b}; categoria={a}. {q}",
    "Para {w} sabemos que a categoria é {a}, o contexto é {c} e a propriedade é {b}. {q}",
)
HELD_QUESTIONS = (
    "Que valor tem a {k} de {w}? Resposta: ",
    "Em relação a {w}, qual é a {k}? Resposta: ",
)

STAGES = ("copy", "select", "language")

def invented(rng, length=5):
    return "".join(rng.choice(ALPHABET) for _ in range(length))

def novel_record(rng):
    # Randomly generated combinations, not a finite memorization table.
    vals = [invented(rng, rng.randint(4, 7)) for _ in range(4)]
    return vals[0], tuple(vals[1:])

def prompt(w, rels, kind, fmt, qfmt):
    a, b, c = rels
    q = qfmt.format(w=w, k=kind)
    return fmt.format(w=w, a=a, b=b, c=c, q=q)

def encode_pair(tok, p, answer):
    pre = tok.encode(p, bos=True)
    seq = pre + tok.encode(answer, eos=True)
    if len(seq) > CONFIG["context"] + 1:
        raise ValueError("Prompt exceeds model context; shorten the templates")
    return seq, len(pre)

def example(tok, rng, mode="dynamic", held=False, stage="language"):
    if mode == "known":
        w, rels = rng.choice(tuple(TRAIN.items()))
    else:
        w, rels = novel_record(rng)
    kind = rng.choice(KINDS)
    if stage == "copy":
        return "Copia exatamente: " + rels[KINDS.index(kind)] + "\\nResposta: ", rels[KINDS.index(kind)]
    if stage == "select":
        a, b, c = rels
        return (f"categoria={a}; propriedade={b}; contexto={c}. "
                f"Pedido: {kind}. Resposta: "), rels[KINDS.index(kind)]
    fmts = HELD_FORMATS if held else FORMATS
    qs = HELD_QUESTIONS if held else QUESTIONS
    p = prompt(w, rels, kind, rng.choice(fmts), rng.choice(qs))
    return p, rels[KINDS.index(kind)]

def loss_for(model, seq, prefix):
    x = torch.tensor(seq[:-1], dtype=torch.long)[None]
    y = torch.tensor(seq[1:], dtype=torch.long)
    y[:prefix-1] = -100
    logits = model(x)[0]
    return F.cross_entropy(logits, y, ignore_index=-100)

@torch.inference_mode()
def generate(model, tok, p, max_tokens=48):
    model.eval()
    ids = tok.encode(p, bos=True)
    out = []
    for _ in range(max_tokens):
        logits = model(torch.tensor(ids[-model.context:], dtype=torch.long)[None])[0, -1].clone()
        logits[0] = logits[1] = logits[3] = -float("inf")
        if not out:
            logits[2] = -float("inf")
        nxt = int(logits.argmax())
        if nxt == 2:
            break
        out.append(nxt)
        ids.append(nxt)
    return tok.decode(out).strip()

def make_eval(seed, n, held=False, real=False, stage="language"):
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        if real:
            w, rels = rng.choice(tuple(TEST.items()))
            kind = rng.choice(KINDS)
            fmts = HELD_FORMATS if held else FORMATS
            qs = HELD_QUESTIONS if held else QUESTIONS
            ans = rels[KINDS.index(kind)]
            if stage == "copy":
                p = "Copia exatamente: " + ans + "\\nResposta: "
            elif stage == "select":
                p = f"categoria={rels[0]}; propriedade={rels[1]}; contexto={rels[2]}. Pedido: {kind}. Resposta: "
            else:
                p = prompt(w, rels, kind, rng.choice(fmts), rng.choice(qs))
        else:
            p, ans = example(None, rng, held=held, stage=stage)
        items.append((p, ans))
    return items

def evaluate(model, tok, items):
    correct = 0
    empty = 0
    char_matches = 0
    char_total = 0
    failures = []
    for p, ans in items:
        actual = generate(model, tok, p)
        correct += actual == ans
        char_matches += sum(x == y for x, y in zip(actual, ans))
        char_total += max(len(actual), len(ans))
        empty += not actual
        if actual != ans and len(failures) < 3:
            failures.append({"prompt": p, "expected": ans, "actual": actual})
    return {"correct": correct, "total": len(items),
            "accuracy_pct": round(100 * correct / len(items), 1),
            "empty": empty, "char_accuracy_pct": round(100 * char_matches / max(1, char_total), 1),
            "sample_errors": failures}

def run_evaluation(model, tok, n, stage):
    # Validation and test have disjoint deterministic seeds and templates.
    suites = {
        "validation_dynamic": make_eval(31001, n, stage=stage),
        "heldout_dynamic": make_eval(42002, n, held=True, stage=stage),
        "heldout_real": make_eval(53003, n, held=True, real=True, stage=stage),
    }
    return {name: evaluate(model, tok, data) for name, data in suites.items()}

def save(path, model, optim, rng, step, best, stage):
    state = {"model": model.state_dict(), "optimizer": optim.state_dict(),
             "config": CONFIG, "tokenizer": {"words": []}, "rng": rng.getstate(),
             "step": step, "best_validation": best, "stage": stage}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=STAGES, default="copy")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rate", type=float, default=0.0002)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-size", type=int, default=30)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=Path.home()/".nord-brain"/"curriculum")
    args = ap.parse_args()
    if not 1 <= args.steps <= 100000 or not 1 <= args.batch_size <= 32:
        ap.error("Invalid steps or batch size")
    if not 0 < args.rate <= 0.001 or not 1 <= args.eval_every <= 100000:
        ap.error("Invalid learning rate or evaluation interval")
    if not 10 <= args.eval_size <= 300:
        ap.error("Evaluation size must be between 10 and 300")
    torch.set_num_threads(max(1, min(4, int(os.getenv("NORD_THREADS", "2")))))
    torch.manual_seed(1729)
    tok = Tokenizer([])
    model = Model(len(tok.vocab), **CONFIG)
    optim = torch.optim.AdamW(model.parameters(), lr=args.rate)
    rng = random.Random(1729)
    path = args.data_dir/"vocabulary-v7-1.pt"
    best_path = args.data_dir/"vocabulary-v7-1-best.pt"
    step, best = 0, -1.
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["config"] != CONFIG or state["tokenizer"]["words"]:
            ap.error("Incompatible V7 checkpoint")
        previous = state.get("stage", "copy")
        if STAGES.index(args.stage) > STAGES.index(previous) + 1:
            ap.error("Advance one stage at a time")
        if STAGES.index(args.stage) > STAGES.index(previous) and state["best_validation"] < 95:
            ap.error("Previous stage needs at least 95% validation accuracy")
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        for group in optim.param_groups:
            group["lr"] = args.rate
        rng.setstate(state["rng"])
        step, best = state["step"], state["best_validation"] if previous == args.stage else -1.
        print("V7 retomada no passo", step, flush=True)
    elif args.eval_only:
        ap.error("V7 checkpoint does not exist")
    else:
        print("Nova V7:", len(tok.vocab), "tokens,", sum(p.numel() for p in model.parameters()), "parâmetros", flush=True)
    if args.eval_only:
        print(json.dumps(run_evaluation(model, tok, args.eval_size, args.stage), ensure_ascii=False), flush=True)
        return
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.steps):
        model.train()
        optim.zero_grad(set_to_none=True)
        # Keep familiar words in rehearsal; most examples teach transfer.
        pairs = []
        for _ in range(args.batch_size):
            mode = "known" if rng.random() < 0.2 else "dynamic"
            # Rehearse earlier tasks to reduce catastrophic forgetting.
            task = args.stage
            if args.stage == "select" and rng.random() < 0.25:
                task = "copy"
            if args.stage == "language" and rng.random() < 0.35:
                task = rng.choice(("copy", "select"))
            p, ans = example(tok, rng, mode=mode, stage=task)
            pairs.append(encode_pair(tok, p, ans))
        loss = torch.stack([loss_for(model, seq, prefix) for seq, prefix in pairs]).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        step += 1
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": round(float(loss.detach()), 4)}), flush=True)
        if step % args.eval_every == 0 or i == args.steps - 1:
            results = run_evaluation(model, tok, args.eval_size)
            print("AVALIAÇÃO V7:", json.dumps({"step": step, "results": results,
                  "stage_2_unlocked": False}, ensure_ascii=False), flush=True)
            score = results["validation_dynamic"]["accuracy_pct"]
            if score > best:
                best = score
                save(best_path, model, optim, rng, step, best, args.stage)
            save(path, model, optim, rng, step, best, args.stage)
    print("Checkpoint V7.1:", path, "| melhor validação:", best,
          "| fase 2 bloqueada", flush=True)

if __name__ == "__main__":
    main()
