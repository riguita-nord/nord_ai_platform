"""Import an official Wikimedia Portuguese Wikipedia XML bz2 dump.

Uses a published, fixed ptwiki snapshot/shard instead of the rate-limited
live API. Download is cached. Import is streaming and stops at target size.
The XML contains WIKITEXT, not clean rendered prose; conservative cleaning
filters out complex markup, but is not equivalent to a full wikitext parser.
"""
from __future__ import annotations
import argparse
import bz2
import fcntl
import hashlib
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import Request, urlopen

SNAPSHOT = "20261101"
FILE = "ptwiki-20261101-pages-articles-multistream1.xml-p1p105695.bz2"
URL = f"https://dumps.wikimedia.org/ptwiki/{SNAPSHOT}/{FILE}"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
TAG = re.compile(r"<[^>]+>")
LINK = re.compile(r"\[\[(?:[^]|]*\|)?([^]|]+)\]\]")
EXT = re.compile(r"\[https?://[^\s\]]+(?:\s+([^\]]+))?\]")
TEMPLATES = re.compile(r"\{\{[^{}]*\}\}")
BADLINE = re.compile(r"(?i)^\s*(?:\||\{\||\|\}|!|category:|ficheiro:|imagem:|file:|image:)")
def clean(wikitext):
    text = re.sub(r"(?s)<!--.*?-->", " ", wikitext)
    text = re.sub(r"(?s)<ref\b[^>]*>.*?</ref\s*>", " ", text, flags=re.I)
    text = re.sub(r"(?s)<(?:gallery|math|code|syntaxhighlight)\b[^>]*>.*?</(?:gallery|math|code|syntaxhighlight)\s*>", " ", text, flags=re.I)
    for _ in range(5):
        text = TEMPLATES.sub(" ", text)
    text = LINK.sub(r"\1", text)
    text = EXT.sub(lambda m: m.group(1) or " ", text)
    text = TAG.sub(" ", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"(?m)^=+\s*(.*?)\s*=+\s*$", r"\n\1\n", text)
    lines = []
    for line in text.splitlines():
        line = line.strip(" *#;: \t")
        if not line or BADLINE.search(line) or "{{" in line or "}}" in line or "{|" in line:
            continue
        if len(line) > 1500:
            continue
        lines.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(text) < 450 or len(text.split()) < 90:
        return ""
    return text[:120000]

def download(cache):
    if cache.is_file() and cache.stat().st_size > 1000000:
        print(f"Using cached Wikimedia dump: {cache} ({cache.stat().st_size} bytes)", flush=True)
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(cache.suffix + ".part")
    print(f"Downloading official dump {URL} (about 205 MB compressed)", flush=True)
    req = Request(URL, headers={"User-Agent": "NordBrainDumpImporter/1.0 (local research)"})
    with urlopen(req, timeout=120) as response, partial.open("wb") as out:
        if response.status != 200:
            raise RuntimeError(f"Unexpected HTTP status: {response.status}")
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > 350_000_000:
                raise RuntimeError("Dump exceeds 350 MB compressed")
            out.write(chunk)
            if total % (20 * 1024 * 1024) < len(chunk):
                print(f"Downloaded {total / 1e6:.0f} MB", flush=True)
    partial.replace(cache)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-mb", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("training/dataset"))
    p.add_argument("--cache", type=Path, default=Path("training/dumps") / FILE)
    p.add_argument("--accept-cc-by-sa", action="store_true")
    args = p.parse_args()
    if not args.accept_cc_by_sa:
        p.error("You must acknowledge CC BY-SA attribution/share-alike requirements")
    if not 1 <= args.target_mb <= 400:
        p.error("target-mb must be between 1 and 400")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".collection.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Já existe outra recolha ativa")
        download(args.cache)
        trainpath, evalpath, sourcepath = out / "train.txt", out / "eval.txt", out / "sources.jsonl"
        seen = set()
        if sourcepath.exists():
            for line in sourcepath.read_text(encoding="utf-8").splitlines():
                try:
                    seen.add(json.loads(line)["sha256"])
                except (KeyError, ValueError):
                    pass
        total = sum(f.stat().st_size for f in (trainpath, evalpath) if f.exists())
        target = args.target_mb * 1000000
        if total >= target:
            print(f"Dataset already meets target: {total} bytes", flush=True)
            return
        accepted = 0
        print("Streaming compressed XML and extracting article text...", flush=True)
        with bz2.open(args.cache, "rb") as stream, \
             trainpath.open("a", encoding="utf-8") as train, \
             evalpath.open("a", encoding="utf-8") as evaluation, \
             sourcepath.open("a", encoding="utf-8") as manifest:
            context = ET.iterparse(stream, events=("end",))
            for _, elem in context:
                if not elem.tag.endswith("}page"):
                    continue
                def child(name):
                    return elem.find("{*}" + name)
                ns = child("ns")
                title = child("title")
                ident = child("id")
                rev = child("revision")
                if ns is None or ns.text != "0" or title is None or ident is None or rev is None:
                    elem.clear()
                    continue
                raw = rev.find("{*}text")
                body = clean(raw.text or "") if raw is not None else ""
                if body:
                    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
                    if digest not in seen:
                        split = "eval" if int(digest[:8], 16) % 20 == 0 else "train"
                        data = body + "\n\n"
                        (evaluation if split == "eval" else train).write(data)
                        manifest.write(json.dumps({
                            "title": title.text, "page_id": ident.text,
                            "url": "https://pt.wikipedia.org/?curid=" + ident.text,
                            "license": "CC BY-SA 4.0", "license_url": LICENSE,
                            "source": "Wikimedia official ptwiki XML dump",
                            "dump_url": URL, "snapshot": SNAPSHOT,
                            "language": "pt-mixed", "sha256": digest, "split": split
                        }, ensure_ascii=False) + "\n")
                        seen.add(digest)
                        total += len(data.encode("utf-8"))
                        accepted += 1
                        if accepted % 100 == 0:
                            train.flush(); evaluation.flush(); manifest.flush()
                            print(f"Articles={accepted} dataset_bytes={total}", flush=True)
                elem.clear()
                if total >= target:
                    break
            train.flush(); evaluation.flush(); manifest.flush()
        print(f"Dump import finished: articles={accepted}, total_bytes={total}", flush=True)
        if total < target:
            print("WARNING: this shard was exhausted before reaching target", flush=True)

if __name__ == "__main__":
    main()
