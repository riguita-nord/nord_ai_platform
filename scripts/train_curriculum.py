"""Four explicit Nord Brain stages: vocabulary, sentences, reasoning, final exam.

Use --stage words|sentences|reasoning|final. No automatic promotion.
Independent checkpoint from previous curriculum prototype (curriculum-four.pt).
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from train_language_v2 import Model, Tokenizer, PATTERN

STAGES = ("words", "sentences", "reasoning", "final")
WORD_RE = re.compile(r"^[^\W\d_]{3,}$", re.UNICODE)
# Reserved arithmetic problems use operands outside the training range.
TEST_MATH = [(31, 12), (24, 15), (42, 16), (53, 21), (17, 27)]


def vocabulary(corpus, limit=1200):
    counts = Counter(p.lower() for p in PATTERN.findall(corpus) if WORD_RE.fullmatch(p))
    return [w for w, n in counts.most_common() if n >= 2][:limit]


def batch_word(tok, words, rng):
    # The target is a complete word; this stage never asks the model to write sentences.
    word = rng.choice(words)
    seq = tok.encode(word, bos=True, eos=True)
    return seq, 0


def batch_sentence(tok, sentences, rng):
    line = rng.choice(sentences)
    seq = tok.encode(line, bos=True, eos=True)
    return seq, 0


def batch_math(tok, rng):
    a, b = rng.randrange(0, 20), rng.randrange(0, 20)
    op = rng.choice(("+", "-"))
    result = a + b if op == "+" else a - b
    prefix = tok.encode(f"Calcula: {a} {op} {b} = ", bos=True)
    suffix = tok.encode(str(result), eos=True)
    return prefix + suffix, len(prefix)


def step_loss(model, seq, prefix=0):
    seq = seq[-model.context - 1:]
    labels = torch.tensor(seq[1:], dtype=torch.long)
    # Mask instruction tokens, but keep the answer targets.
    labels[:max(0, prefix - 1)] = -100
    if not (labels != -100).any():
        return None
    return F.cross_entropy(model(torch.tensor(seq[:-1], dtype=torch.long)[None])[0],
                           labels, ignore_index=-100)


@torch.inference_mode()
def complete(model, tok, prompt, max_tokens=32):
    model.eval()
    ids = tok.encode(prompt, bos=True)
    output = []
    for _ in range(max_tokens):
        logits = model(torch.tensor(ids[-model.context:], dtype=torch.long)[None])[0, -1].clone()
        logits[0] = logits[1] = logits[3] = -float("inf")
        nxt = int(logits.argmax())
        if nxt == 2:
            break
        output.append(nxt)
        ids.append(nxt)
    return tok.decode(output).strip()


def evaluate_words(model, tok, words):
    # Score each corpus word under the neural model, not merely the tokenizer.
    # This is a ranking of learned word likelihood, not proof of semantic knowledge.
    model.eval()
    ranked = []
    with torch.inference_mode():
        for word in words[:250]:
            seq = tok.encode(word, bos=True, eos=True)
            loss = step_loss(model, seq)
            if loss is not None:
                ranked.append({"word": word, "mean_token_loss": round(float(loss), 4)})
    ranked.sort(key=lambda p: p["mean_token_loss"])
    return ranked[:30]


def evaluate_sentences(model, tok):
    prompts = ("A menina abriu ", "Hoje de manhã ", "O livro estava ")
    return [{"prompt": p, "generated": complete(model, tok, p, 45)} for p in prompts]


def evaluate_math(model, tok):
    results = []
    for a, b in TEST_MATH:
        prompt = f"Calcula: {a} + {b} = "
        generated = complete(model, tok, prompt, 10)
        results.append({"question": prompt.strip(), "generated": generated,
                        "expected": str(a+b), "correct": generated == str(a+b)})
    return results


def report(model, tok, words, stage):
    if stage == "words":
        ranked = evaluate_words(model, tok, words)
        print("VOCABULÁRIO — 30 palavras com maior probabilidade no modelo (entre 250 candidatas):", flush=True)
        print("Isto mede probabilidade, não compreensão do significado.", flush=True)
        for p in ranked:
            print(json.dumps(p, ensure_ascii=False), flush=True)
    elif stage == "sentences":
        print("FRASES — exemplos gerados:", flush=True)
        for p in evaluate_sentences(model, tok):
            print(json.dumps(p, ensure_ascii=False), flush=True)
    elif stage == "reasoning":
        print("RACIOCÍNIO — exercícios não utilizados no treino:", flush=True)
        for p in evaluate_math(model, tok):
            print(json.dumps(p, ensure_ascii=False), flush=True)
    else:
        for section in ("words", "sentences", "reasoning"):
            report(model, tok, words, section)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("training/dataset/train.txt"))
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--data-dir", type=Path, default=Path.home() / ".nord-brain" / "curriculum")
    ap.add_argument("--rate", type=float, default=0.0002)
    ap.add_argument("--preview", action="store_true", help="Accepted for compatibility; reports always print")
    args = ap.parse_args()
    if not 1 <= args.steps <= 100000 or not 0 < args.rate <= .001:
        ap.error("Invalid steps or learning rate")
    torch.set_num_threads(max(1, min(4, int(os.getenv("NORD_THREADS", "2")))))
    rng = random.Random(1729)
    torch.manual_seed(1729)
    corpus = args.corpus.read_text(encoding="utf-8")
    if len(corpus.encode("utf-8")) < 100000:
        ap.error("Corpus must be at least 100 KB")
    words = vocabulary(corpus)
    if len(words) < 100:
        ap.error("Insufficient vocabulary")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", corpus)
                 if 20 <= len(s.strip()) <= 300]
    if len(sentences) < 20:
        ap.error("Insufficient sentences")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    path = args.data_dir / "curriculum-four.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        tok = Tokenizer(state["tokenizer"]["words"])
        model = Model(len(tok.vocab), **state["config"])
        model.load_state_dict(state["model"])
        optim = torch.optim.AdamW(model.parameters(), lr=args.rate)
        optim.load_state_dict(state["optimizer"])
        for group in optim.param_groups:
            group["lr"] = args.rate
        completed = list(state["completed"])
        steps = dict(state["steps"])
        print("A retomar treino:", json.dumps(steps), flush=True)
    else:
        if args.stage != "words":
            ap.error("Start with --stage words")
        tok = Tokenizer.fit(corpus, 4096)
        config = {"dim": 192, "heads": 6, "layers": 6, "context": 256}
        model = Model(len(tok.vocab), **config)
        optim = torch.optim.AdamW(model.parameters(), lr=args.rate)
        completed = []
        steps = {s: 0 for s in STAGES[:-1]}
        print("Novo modelo:", sum(p.numel() for p in model.parameters()), "parâmetros", flush=True)
    if args.stage == "final":
        if not all(s in completed for s in STAGES[:-1]):
            ap.error("Final evaluation requires words, sentences and reasoning training first")
        report(model, tok, words, "final")
        return
    idx = STAGES.index(args.stage)
    if idx and STAGES[idx-1] not in completed:
        ap.error(f"First train {STAGES[idx-1]}")
    for i in range(args.steps):
        if args.stage == "words":
            seq, prefix = batch_word(tok, words, rng)
        elif args.stage == "sentences":
            seq, prefix = batch_sentence(tok, sentences, rng)
        else:
            seq, prefix = batch_math(tok, rng)
        loss = step_loss(model, seq, prefix)
        if loss is None:
            continue
        model.train()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        steps[args.stage] += 1
        if (i+1) % 50 == 0 or i+1 == args.steps:
            print(json.dumps({"stage": args.stage, "steps": steps[args.stage],
                              "loss": round(loss.detach().item(), 4)}, ensure_ascii=False), flush=True)
    if args.stage not in completed:
        completed.append(args.stage)
    config = {"dim": model.embed.embedding_dim, "heads": model.blocks[0].attn.num_heads,
              "layers": len(model.blocks), "context": model.context}
    state = {"model": model.state_dict(), "optimizer": optim.state_dict(),
             "tokenizer": tok.to_dict(), "config": config,
             "completed": completed, "steps": steps}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    report(model, tok, words, args.stage)
    print("Checkpoint:", path, flush=True)


if __name__ == "__main__":
    main()
