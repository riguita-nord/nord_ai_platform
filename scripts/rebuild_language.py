"""Rebuild a clean, separate Nord conversational checkpoint from approved PT-PT text.

Offline, CPU-only, resumable. No external model or generated labels.
Run while the interactive Nord process is STOPPED.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nord_brain.core import NeuralCore, TransformerConfig
from nord_brain.linguistic_training import TRAIN, DIALOGUE_TRAIN, HOLDOUT
from nord_brain.conversation_corpus import TRAINING as NATURAL_TRAIN
from nord_brain.dialogue_format import training_prompt


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Approved UTF-8 Portuguese .txt file (minimum 1 MB)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=Path.home()/".nord-brain")
    parser.add_argument("--reset", action="store_true",
                        help="Archive previous experimental checkpoint and start from random weights")
    parser.add_argument("--rate", type=float, default=0.0002)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--preview", action="store_true",
                        help="Print raw unseen generations after training")
    return parser.parse_args()


def main():
    args = parse()
    if not 1 <= args.steps <= 1000000 or not 0 < args.rate <= .001:
        raise SystemExit("Invalid steps or learning rate")
    if not 1 <= args.save_every <= 10000:
        raise SystemExit("Invalid checkpoint interval")
    source = args.corpus.expanduser().resolve(strict=True)
    if source.suffix.lower() != ".txt" or not source.is_file():
        raise SystemExit("Corpus must be a UTF-8 .txt file")
    if not 1000000 <= source.stat().st_size <= 500000000:
        raise SystemExit("Corpus must be between 1 MB and 500 MB")
    corpus = source.read_text(encoding="utf-8")
    if len(corpus.strip()) < 100000:
        raise SystemExit("Corpus has too little text")
    root = args.data_dir.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root/"language-v1.pt"
    # The normal Brain runtime should never run concurrently with this trainer.
    import fcntl
    with (root/"language-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another language trainer is running")
        if args.reset and checkpoint.exists():
            archive = root / ("language-v1.backup-" + time.strftime("%Y%m%d-%H%M%S") + ".pt")
            shutil.copy2(checkpoint, archive)
            print(f"Previous checkpoint backed up: {archive}", flush=True)
            checkpoint.unlink()
        # Fresh model never inherits math weights. The legacy model is unchanged.
        if not checkpoint.exists():
            config = TransformerConfig(d_model=128, layers=8, heads=4,
                                       kv_heads=2, intermediate_size=512, context=512)
            core = NeuralCore(checkpoint, config=config)
            core.save()
            print(f"Created clean conversational model ({core.parameter_count} parameters)", flush=True)
        else:
            core = NeuralCore(checkpoint)
            print(f"Resuming language checkpoint at step {core.steps}", flush=True)
        # No train/evaluation leakage: curated HOLDOUT is never sampled.
        examples = tuple(TRAIN) + tuple(NATURAL_TRAIN) + tuple(
            (training_prompt(turns, core.model.config.context), answer)
            for turns, answer in DIALOGUE_TRAIN)
        if not examples:
            raise SystemExit("No dialogue examples")
        initial = core.score(HOLDOUT)
        print(f"Holdout loss before: {initial:.4f}", flush=True)
        for i in range(args.steps):
            step = core.steps
            if i % 4 == 0:
                # Sample a different region of the full corpus every update.
                size = min(1400, len(corpus))
                start = (step * 7919) % max(1, len(corpus) - size + 1)
                segment = corpus[start:start + size]
                loss = core.train_text(segment, updates=1, rate=args.rate)
            else:
                question, answer = examples[(step * 37 + step // 11) % len(examples)]
                q = question if question.startswith("Tu: ") else "Tu: " + question
                loss = core.train(q, answer, epochs=1, rate=args.rate, dialogue=True)
            if not math.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            if (i + 1) % args.save_every == 0 or i + 1 == args.steps:
                core.save()
                score = core.score(HOLDOUT)
                print(json.dumps({"step": core.steps, "train_loss": round(loss, 4),
                                  "holdout_loss": round(score, 4)}, ensure_ascii=False), flush=True)
        if args.preview:
            for prompt in ("Tu: Olá", "Tu: Como estás hoje?",
                           "Tu: Podemos conversar sobre uma ideia nova?"):
                answer = core.generate(prompt, max_bytes=140, temperature=0, dialogue=True)
                print(f"PREVIEW {prompt!r}: {answer!r}", flush=True)
        print("Training finished. Test actual unseen conversations before treating the model as ready.", flush=True)


if __name__ == "__main__":
    main()
