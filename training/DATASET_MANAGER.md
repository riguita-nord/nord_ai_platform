# Nord Dataset Manager (experimental)

The dataset manager collects text from Portuguese Wikipedia, without third-party
Python dependencies or external AI models. **Wikipedia Portuguese is mixed PT-PT
and PT-BR**, not a certified European Portuguese corpus. It is also encyclopedia
text, not conversational instruction data.

## Download (network access required)

Stop the interactive Nord process before training. From the repository root:

    git pull origin main
    .venv/bin/python scripts/dataset_manager.py --target-mb 10 --accept-cc-by-sa

Outputs:

- training/dataset/train.txt: model training material
- training/dataset/eval.txt: separate document-level evaluation material
- training/dataset/sources.jsonl: per-document attribution, revision and license
- training/dataset/README-LICENSE.txt: license information

The Wikipedia material is licensed CC BY-SA 4.0. Review the license before use,
and preserve the manifest and attribution when sharing any dataset or derivative.
Do not upload the dataset to GitHub accidentally. The downloader is resumable,
deduplicates exact cleaned text, and rate limits API requests. If Wikipedia
blocks API requests or does not provide extracts, the tool reports the error.

## Train a clean experimental language checkpoint (CPU)

Check that train.txt is at least 1 MB:

    wc -c training/dataset/train.txt

Then run:

    .venv/bin/python scripts/rebuild_language.py \
      --corpus training/dataset/train.txt --reset \
      --steps 1000 --save-every 50 --preview

**Only use --reset for the first clean run.** It backs up the previous
language-v1.pt checkpoint before replacing it. For later sessions omit --reset.
The checkpoint remains under ~/.nord-brain by default. Do not run ./nord
concurrently with the trainer.

The model still uses byte-level tokens and eight Transformer layers. A dataset
of 10 MB and 1000 steps is an initial experiment, not sufficient evidence of
natural conversational fluency. The training script's heldout loss is based
on its existing curated holdout, and the Wikipedia eval.txt is reserved for
future corpus-specific evaluation. Inspect the --preview generations before
claiming a quality improvement.

No source data is bundled by this change; download requires internet access.
