# Nord Brain V3 — CPU experimental

This experiment is separate from V1 and V2. It uses the existing V2 hybrid word/UTF-8-byte tokenizer as a **transitional baseline**, not BPE. A real BPE tokenizer and larger dialogue dataset are **not yet implemented**.

Run from repository root:

```bash
git pull origin main
.venv/bin/python -m compileall -q scripts/train_language_v3.py
NORD_THREADS=2 .venv/bin/python scripts/train_language_v3.py --corpus training/dataset/train.txt --stage language --steps 100 --preview
```

If the first run completes, continue the language phase before dialogue:

```bash
NORD_THREADS=2 .venv/bin/python scripts/train_language_v3.py --stage language --steps 2000 --preview
NORD_THREADS=2 .venv/bin/python scripts/train_language_v3.py --stage dialogue --steps 100 --preview
```

The dialogue phase uses 80% supervised dialogue updates and 20% text rehearsal. Both phases resume the same V3 checkpoint at `~/.nord-brain/v3/language-v3.pt`. The `--eval-only` option compares sampled and greedy decoding without modifying the checkpoint.

Default: 3.5M parameters, context 256, 2 CPU threads. You can start a *new* experiment with `--dim 256 --heads 8 --layers 6 --data-dir ~/.nord-brain/v3-large`, which is slower and consumes more RAM. Architecture flags do not resize existing checkpoints.

The holdout metric is dialogue next-token loss, not a fluency guarantee. A falling loss alone does not mean the model can converse. This implementation has not been benchmarked on the user's VM. For serious language learning, a substantially larger licensed PT-PT dataset, independent validation, and a more efficient subword tokenizer are still required.
