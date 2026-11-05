# Nord Brain: multi-source deep-learning data pipeline

The Nord language model is a **real PyTorch Transformer**, trained with
gradient descent (next-token prediction). It runs locally on CPU. No external
chatbot, API-generated labels, or Qwen model is used.

## 1. Collect Wikipedia articles (network)

    .venv/bin/python scripts/dataset_manager.py --target-mb 10 \
      --delay 10 --accept-cc-by-sa

The Wikipedia API may impose rate limits. Keep the source manifest. Wikipedia
Portuguese is **mixed pt-PT and pt-BR**, not pure European Portuguese.

## 2. Add other websites and text collections (approved exports)

Download/export licensed **plain UTF-8 text** from the original publisher
yourself, checking the license and terms. Do not indiscriminately scrape
copyrighted sites. For each document add one JSON object per line to
training/approved_sources.jsonl:

    {"path":"./texts/example.txt","source_url":"https://example.org/record/1","license":"CC-BY-4.0","language":"pt-PT","title":"Example"}

Allowed licenses: CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0, PUBLIC-DOMAIN.
These labels are supplied by you; the importer **does not independently
verify copyright**. Keep original attribution and follow license obligations.
The path is relative to the JSONL manifest file. Each file should be one
document, not a mixture of unrelated sources.

Import:

    .venv/bin/python scripts/import_licensed_texts.py \
      --manifest training/approved_sources.jsonl

Both ingestion tools write to training/dataset/train.txt and eval.txt,
with a common sources.jsonl provenance file. Exact duplicate documents are
excluded and each document is assigned to a stable train/eval split. This
is **not** near-duplicate detection or a guarantee against source leakage.

## 3. Train locally (CPU)

Stop ./nord before running the trainer. Check the size:

    wc -c training/dataset/train.txt

At least 1,000,000 bytes are required. On the first clean run:

    .venv/bin/python scripts/rebuild_language.py \
      --corpus training/dataset/train.txt \
      --reset --steps 1000 --save-every 50 --text-ratio 0.75 --preview

On subsequent runs, **omit --reset** to resume from the existing checkpoint.
The trainer uses 75% of updates for text prediction and 25% for curated
dialogue examples, and saves checkpoints every 50 updates. Its existing
heldout dialogue loss and raw generation previews should both be monitored.
Do not run multiple trainers or ./nord simultaneously.

## Limitations

This is **not** a web-browsing AI, an autonomous internet-learning agent,
or a pretrained conversational LLM. The model still has eight layers and a
byte tokenizer; 10 MB of text and 1000 CPU steps are an experiment, not a
guarantee of fluency. Encyclopedic text alone does not teach turn-taking,
instruction-following, or safe factual attribution. A future iteration needs
better tokenizer, larger licensed corpus, conversational supervision, and
genuine evaluation of unseen responses.
