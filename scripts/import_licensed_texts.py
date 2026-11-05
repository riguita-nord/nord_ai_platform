"""Import approved UTF-8 text documents from multiple licensed sources.

Manifest format (JSONL):
{"path":"./texts/book.txt","source_url":"https://example.org/book",
 "license":"CC0-1.0","language":"pt-PT","title":"Book"}

No web scraping or automatic license assumptions. Source files are local,
allowing official downloads/exports from Wikipedia, Wikisource, universities,
or other rights-cleared collections. Only plain .txt files are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SPACE = re.compile(r"[ \t]+")
PARAGRAPHS = re.compile(r"\n{3,}")
ALLOWED_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "PUBLIC-DOMAIN"}
ALLOWED_LANGUAGES = {"pt-PT", "pt-mixed", "pt-BR"}


def normalized(text):
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE.sub(" ", text)
    return PARAGRAPHS.sub("\n\n", text).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("training/dataset"))
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error("Manifest JSONL does not exist")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_out = args.output / "sources.jsonl"
    train_path = args.output / "train.txt"
    eval_path = args.output / "eval.txt"
    seen = set()
    if manifest_out.exists():
        for line in manifest_out.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["sha256"])
            except (ValueError, KeyError):
                continue
    imported, skipped = 0, 0
    with args.manifest.open(encoding="utf-8") as source, \
         train_path.open("a", encoding="utf-8") as train, \
         eval_path.open("a", encoding="utf-8") as evaluation, \
         manifest_out.open("a", encoding="utf-8") as provenance:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                name = Path(record["path"]).expanduser()
                if not name.is_absolute():
                    name = (args.manifest.parent / name).resolve()
                license_id = record["license"]
                language = record["language"]
                url = record["source_url"]
                title = record["title"]
                if license_id not in ALLOWED_LICENSES or language not in ALLOWED_LANGUAGES:
                    raise ValueError("unsupported license or language")
                if not isinstance(url, str) or not url.startswith("https://"):
                    raise ValueError("source_url must be an HTTPS URL")
                if not isinstance(title, str) or not title.strip():
                    raise ValueError("title is required")
                if name.suffix.lower() != ".txt" or not name.is_file():
                    raise ValueError("path must reference an existing .txt file")
                if name.stat().st_size > 100_000_000:
                    raise ValueError("file exceeds 100 MB; split it into smaller documents")
                body = normalized(name.read_text(encoding="utf-8"))
                if len(body) < 450:
                    raise ValueError("text is too short")
                digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
                if digest in seen:
                    skipped += 1
                    continue
                split = "eval" if int(digest[:8], 16) % 20 == 0 else "train"
                target = evaluation if split == "eval" else train
                target.write(body + "\n\n")
                provenance.write(json.dumps({
                    "title": title, "url": url, "license": license_id,
                    "language": language, "sha256": digest, "split": split,
                    "source": "user-approved-local-text",
                }, ensure_ascii=False) + "\n")
                target.flush()
                provenance.flush()
                seen.add(digest)
                imported += 1
            except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
                skipped += 1
                print(f"Skipped manifest line {number}: {exc}", file=sys.stderr)
    print(f"Imported={imported}, skipped={skipped}, "
          f"train_bytes={train_path.stat().st_size}, eval_bytes={eval_path.stat().st_size}")


if __name__ == "__main__":
    main()
