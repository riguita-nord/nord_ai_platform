"""Nord Brain V3: CPU-friendly progressive language and dialogue training.

Uses V2's validated word/byte tokenizer as a transitional baseline.
V3 checkpoints are isolated from V1 and V2; no external model or API.
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
from train_language_v2 import Tokenizer, Model, evaluate
from nord_brain.linguistic_training import TRAIN, DIALOGUE_TRAIN, HOLDOUT
from nord_brain.conversation_corpus import TRAINING as NATURAL_TRAIN
from nord_brain.dialogue_format import training_prompt


@torch.inference_mode()
def sample(model, tokenizer, question, *, temperature=.75, top_p=.9, repeat_penalty=1.18, limit=100):
    model.eval()
    ids = tokenizer.encode("Tu: " + question + "\nNord: ", bos=True)
    generated = []
    for _ in range(limit):
        logits = model(torch.tensor(ids[-model.context:], dtype=torch.long)[None])[0, -1].clone()
        logits[0] = -float("inf")
        logits[1] = -float("inf")
        logits[3] = -float("inf")
        recent = set(generated[-40:])
        for token in recent:
            if token != 2:
                logits[token] = logits[token] / repeat_penalty if logits[token] > 0 else logits[token] * repeat_penalty
        if temperature == 0:
            token = int(logits.argmax())
        else:
            logits = logits / temperature
            sorted_logits, sorted_idx = logits.sort(descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = probs.cumsum(-1)
            remove = (cumulative - probs) >= top_p
            sorted_logits[remove] = -float("inf")
            probs = F.softmax(sorted_logits, dim=-1)
            token = int(sorted_idx[torch.multinomial(probs, 1)])
        if token == 2:
            break
        ids.append(token)
        generated.append(token)
        if "\nTu:" in tokenizer.decode(generated):
            break
    return tokenizer.decode(generated).split("\nTu:")[0].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("training/dataset/train.txt"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--stage", choices=("language", "dialogue"), default="language")
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--rate", type=float, default=0.0002)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--data-dir", type=Path, default=Path.home() / ".nord-brain" / "v3")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.steps <= 100000 or not 128 <= args.context <= 512:
        ap.error("Steps/context out of range")
    if not 512 <= args.vocab <= 8192 or not 0 < args.rate <= .001:
        ap.error("Vocab/rate out of range")
    if args.dim % args.heads or not 96 <= args.dim <= 384 or not 2 <= args.layers <= 10:
        ap.error("Model dimensions invalid")
    torch.set_num_threads(max(1, min(4, int(os.getenv("NORD_THREADS", "2")))))
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    path = args.data_dir / "language-v3.pt"
    corpus = args.corpus.read_text(encoding="utf-8")
    if len(corpus.encode("utf-8")) < 100000:
        ap.error("At least 100 KB of approved text is required for a smoke test")
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        tokenizer = Tokenizer(state["tokenizer"]["words"])
        config = state["config"]
        model = Model(len(tokenizer.vocab), **config)
        model.load_state_dict(state["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate)
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.rate
        step = int(state["step"])
        language_steps = int(state.get("language_steps", 0))
        dialogue_steps = int(state.get("dialogue_steps", 0))
        print("Resuming V3 at", step, "steps", flush=True)
    else:
        if args.stage == "dialogue":
            ap.error("Start with --stage language before dialogue")
        tokenizer = Tokenizer.fit(corpus, args.vocab)
        config = dict(dim=args.dim, layers=args.layers, heads=args.heads, context=args.context)
        model = Model(len(tokenizer.vocab), **config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate)
        step = language_steps = dialogue_steps = 0
        print("New V3:", sum(p.numel() for p in model.parameters()), "parameters", flush=True)
    print("Holdout before:", round(evaluate(model, tokenizer, HOLDOUT), 4), flush=True)
    text_ids = tokenizer.encode(corpus)
    if len(text_ids) <= config["context"]:
        ap.error("Corpus too small for context window")
    dialogue = list(TRAIN) + list(NATURAL_TRAIN) + [
        (training_prompt(turns, config["context"]), answer) for turns, answer in DIALOGUE_TRAIN
    ]
    rng.shuffle(dialogue)
    for i in range(0 if args.eval_only else args.steps):
        # In dialogue stage, rehearse language on 20% of updates to reduce forgetting.
        do_language = args.stage == "language" or (i % 5 == 0)
        if do_language:
            start = rng.randrange(len(text_ids) - config["context"])
            sequence = text_ids[start:start + config["context"] + 1]
            targets = torch.tensor(sequence[1:], dtype=torch.long)
            language_steps += 1
        else:
            question, answer = dialogue[rng.randrange(len(dialogue))]
            prefix = tokenizer.encode((question if question.startswith("Tu: ") else "Tu: " + question) + "\nNord: ", bos=True)
            suffix = tokenizer.encode(answer, eos=True)
            sequence = (prefix + suffix)[-config["context"] - 1:]
            targets = torch.tensor(sequence[1:], dtype=torch.long)
            masked = max(0, len(prefix) - (len(prefix) + len(suffix) - len(sequence)) - 1)
            targets[:masked] = -100
            dialogue_steps += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(torch.tensor(sequence[:-1], dtype=torch.long)[None])[0]
        loss = F.cross_entropy(logits, targets, ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if (i + 1) % 50 == 0 or i + 1 == args.steps:
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "tokenizer": tokenizer.to_dict(), "config": config, "step": step,
                     "language_steps": language_steps, "dialogue_steps": dialogue_steps}
            tmp = path.with_suffix(".tmp")
            torch.save(state, tmp)
            os.replace(tmp, path)
            print(json.dumps({"step": step, "stage": args.stage,
                              "language_steps": language_steps, "dialogue_steps": dialogue_steps,
                              "train_loss": round(loss.detach().item(), 4),
                              "dialogue_holdout_loss": round(evaluate(model, tokenizer, HOLDOUT), 4)}),
                  flush=True)
    if args.preview or args.eval_only:
        for prompt in ("Olá", "Como estás hoje?", "Podemos conversar sobre uma ideia nova?"):
            print("PREVIEW", repr(prompt), "sampled", repr(sample(model, tokenizer, prompt)),
                  "greedy", repr(sample(model, tokenizer, prompt, temperature=0)), flush=True)


if __name__ == "__main__":
    main()
