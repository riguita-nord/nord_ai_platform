"""Nord Dataset Manager: download, clean and split attributed Portuguese Wikipedia text.

Wikipedia is mixed PT-PT/PT-BR; this tool does not falsely label it PT-PT.
Wikipedia text is CC BY-SA 4.0; attribution and share-alike obligations apply.
No dependencies beyond Python's standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://pt.wikipedia.org/w/api.php"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
USER_AGENT = "NordBrainDatasetManager/1.0 (research; attribution stored; local training)"
SPACE = re.compile(r"[ \t]+")
BLOCK = re.compile(r"\n{3,}")
BAD = re.compile(r"(?i)(?:\{\{|\}\}|<script|javascript:|https?://\S{150,})")


def fetch(params, attempts=4):
    url = API + "?" + urlencode(params)
    for n in range(attempts):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urlopen(req, timeout=35) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code == 429:
                raw = exc.headers.get("Retry-After", "") if exc.headers else ""
                wait = f" Retry-After={raw}s." if raw else ""
                raise RuntimeError(
                    "Wikipedia devolveu HTTP 429 (limite de pedidos)." + wait +
                    " Recolha interrompida sem perder os artigos guardados. "
                    "Tenta mais tarde ou utiliza um dump oficial da Wikimedia."
                ) from exc
            if n == attempts - 1 or exc.code in (400, 401, 403, 404):
                raise RuntimeError(f"Wikipedia API unavailable: {exc}") from exc
            time.sleep(min(30, 3 * (n + 1)))
        except (URLError, TimeoutError, ValueError) as exc:
            if n == attempts - 1:
                raise RuntimeError(f"Wikipedia API unavailable: {exc}") from exc
            time.sleep(min(30, 3 * (n + 1)))
    raise RuntimeError("Unreachable")


def clean(value):
    value = html.unescape(value or "")
    value = value.replace("\r", "\n").replace("\x00", "")
    value = SPACE.sub(" ", value)
    value = BLOCK.sub("\n\n", value).strip()
    if len(value) < 450 or BAD.search(value):
        return ""
    if len(value.split()) < 90:
        return ""
    return value[:120000]


def pages_batch(batch=50):
    data = fetch({"action": "query", "format": "json", "formatversion": "2",
                  "generator": "random", "grnnamespace": "0", "grnlimit": str(batch),
                  "prop": "extracts|info|revisions", "explaintext": "1",
                  "exsectionformat": "plain", "inprop": "url",
                  "rvprop": "ids|timestamp", "redirects": "1"})
    if "error" in data:
        raise RuntimeError(f"Wikipedia API error: {data['error']}")
    return data.get("query", {}).get("pages", [])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("training/dataset"))
    p.add_argument("--target-mb", type=int, default=10)
    p.add_argument("--max-requests", type=int, default=5000)
    p.add_argument("--delay", type=float, default=5.0,
                   help="Seconds between API requests (minimum 3)")
    p.add_argument("--accept-cc-by-sa", action="store_true",
                   help="Acknowledge CC BY-SA 4.0 attribution/share-alike obligations")
    args = p.parse_args()
    if not args.accept_cc_by_sa:
        p.error("Wikipedia content requires explicit --accept-cc-by-sa")
    if not 1 <= args.target_mb <= 400 or not 1 <= args.max_requests <= 100000 or not 3 <= args.delay <= 3600:
        p.error("Invalid target size or request count")
    out = args.output.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    train_path, eval_path = out / "train.txt", out / "eval.txt"
    manifest_path = out / "sources.jsonl"
    existing = set()
    bytes_written = 0
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                existing.add(json.loads(line)["sha256"])
            except (ValueError, KeyError):
                continue
    for path in (train_path, eval_path):
        if path.exists():
            bytes_written += path.stat().st_size
    target = args.target_mb * 1000000
    accepted = 0
    with train_path.open("a", encoding="utf-8") as train, \
         eval_path.open("a", encoding="utf-8") as evaluation, \
         manifest_path.open("a", encoding="utf-8") as manifest:
        for n in range(args.max_requests):
            if bytes_written >= target:
                break
            try:
                pages = pages_batch()
            except RuntimeError as exc:
                print(f"Stopped: {exc}", file=sys.stderr)
                break
            for page in pages:
                title = page.get("title", "")
                body = clean(page.get("extract", ""))
                if not body or page.get("missing"):
                    continue
                digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
                if digest in existing:
                    continue
                existing.add(digest)
                page_id = page.get("pageid")
                if not isinstance(page_id, int):
                    continue
                revisions = page.get("revisions") or [{}]
                record = {
                    "title": title, "page_id": page_id,
                    "url": page.get("fullurl", f"https://pt.wikipedia.org/?curid={page_id}"),
                    "revision_id": revisions[0].get("revid"),
                    "revision_timestamp": revisions[0].get("timestamp"),
                    "license": "CC BY-SA 4.0", "license_url": LICENSE,
                    "source": "Portuguese Wikipedia", "language": "pt-mixed",
                    "sha256": digest,
                    "split": "eval" if int(digest[:8], 16) % 20 == 0 else "train",
                }
                # The stable document hash prevents same-document train/eval leakage.
                target_file = evaluation if record["split"] == "eval" else train
                text = body + "\n\n"
                target_file.write(text)
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                bytes_written += len(text.encode("utf-8"))
                accepted += 1
                if bytes_written >= target:
                    break
            train.flush()
            evaluation.flush()
            manifest.flush()
            if (n + 1) % 10 == 0:
                print(f"Requests={n+1} pages={accepted} bytes={bytes_written}", flush=True)
            if bytes_written < target:
                time.sleep(args.delay)
    readme = out / "README-LICENSE.txt"
    readme.write_text(
        "Source: Portuguese Wikipedia (mixed PT-PT/PT-BR).\n"
        "License: Creative Commons Attribution-ShareAlike 4.0 International\n"
        f"License URL: {LICENSE}\n"
        "The sources.jsonl manifest records page titles, URLs, revision IDs and timestamps.\n"
        "Retain attribution and comply with CC BY-SA obligations when redistributing.\n"
        "This dataset is NOT verified European Portuguese and is NOT dialogue data.\n",
        encoding="utf-8")
    print(f"Dataset ready: {train_path} ({train_path.stat().st_size} bytes), "
          f"{eval_path} ({eval_path.stat().st_size} bytes). "
          f"Sources: {manifest_path}.", flush=True)
    if train_path.stat().st_size < 1000000:
        print("WARNING: train.txt is under 1 MB; request more pages before training.", flush=True)


if __name__ == "__main__":
    main()
