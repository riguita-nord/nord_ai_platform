"""Download licensed public text pages from explicitly approved URLs.

No automatic copyright claims. Restricts requests to public HTTPS hosts,
does not follow redirects, checks robots.txt, limits download size, and
records document provenance. The operator must verify the stated license.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import html
from html.parser import HTMLParser
import ipaddress
import json
import re
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urljoin
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.robotparser import RobotFileParser

UA = "NordBrainTrainer/1.0 (approved research text collector)"
LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "PUBLIC-DOMAIN"}
OUT = Path("training/dataset")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignore = 0
        self.blocks = {"p", "article", "h1", "h2", "h3", "li", "blockquote"}
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "nav", "footer", "header"):
            self.ignore += 1
        if tag in self.blocks and not self.ignore:
            self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "nav", "footer", "header"):
            self.ignore = max(0, self.ignore - 1)
        if tag in self.blocks and not self.ignore:
            self.parts.append("\n")
    def handle_data(self, data):
        if not self.ignore:
            self.parts.append(data)


def public_url(url):
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.port not in (None, 443):
        raise ValueError("Só são permitidos URLs HTTPS públicos sem credenciais ou portas especiais")
    host = parts.hostname
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("Domínio não resolvido") from exc
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Endereço privado, reservado ou local não permitido")
    return parts


def fetch(url, limit=2_000_000):
    public_url(url)
    # DNS can change between validation and connection (DNS rebinding).
    # Require an explicit host allowlist for deployment, in addition to checks.
    allow = {x.strip().lower() for x in __import__("os").environ.get("NORD_ALLOWED_SOURCE_HOSTS", "").split(",") if x.strip()}
    host = urlsplit(url).hostname.lower()
    if host not in allow:
        raise ValueError("Domínio não autorizado; adiciona-o a NORD_ALLOWED_SOURCE_HOSTS")
    robot_url = "https://" + host + "/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robot_url)
    try:
        rp.read()
    except Exception as exc:
        raise ValueError("Não foi possível verificar robots.txt") from exc
    if not rp.can_fetch(UA, url):
        raise ValueError("robots.txt não permite a recolha")
    opener = build_opener(NoRedirect())
    req = Request(url, headers={"User-Agent": UA, "Accept": "text/plain,text/html"})
    with opener.open(req, timeout=25) as response:
        mime = response.headers.get_content_type()
        if mime not in ("text/plain", "text/html"):
            raise ValueError("A página não contém texto HTML ou TXT")
        if int(response.headers.get("Content-Length", "0")) > limit:
            raise ValueError("Página demasiado grande")
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Página excede o limite de 2 MB")
        encoding = response.headers.get_content_charset() or "utf-8"
        decoded = raw.decode(encoding, errors="replace")
    if mime == "text/html":
        parser = Extract()
        parser.feed(decoded)
        decoded = " ".join(parser.parts)
    return re.sub(r"\s+", " ", html.unescape(decoded)).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sourcefile = args.output / "sources.jsonl"
    seen = set()
    if sourcefile.exists():
        for line in sourcefile.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["sha256"])
            except (KeyError, ValueError):
                pass
    with args.manifest.open(encoding="utf-8") as inputfile, \
         (args.output / "train.txt").open("a", encoding="utf-8") as train, \
         (args.output / "eval.txt").open("a", encoding="utf-8") as evaluation, \
         sourcefile.open("a", encoding="utf-8") as provenance:
        for i, line in enumerate(inputfile, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                url, license_id = record["url"], record["license"]
                if license_id not in LICENSES:
                    raise ValueError("Licença não suportada")
                body = fetch(url)
                if len(body) < 450:
                    raise ValueError("Texto insuficiente")
                digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
                if digest in seen:
                    print(f"Duplicate: {url}", flush=True)
                    continue
                split = "eval" if int(digest[:8], 16) % 20 == 0 else "train"
                (evaluation if split == "eval" else train).write(body + "\n\n")
                provenance.write(json.dumps({"url": url, "title": record.get("title", url),
                    "license": license_id, "language": record.get("language", "pt-mixed"),
                    "sha256": digest, "split": split, "source": "approved-web-url"}, ensure_ascii=False) + "\n")
                train.flush()
                evaluation.flush()
                provenance.flush()
                seen.add(digest)
                print(f"Imported: {url} ({len(body)} chars)", flush=True)
            except (ValueError, OSError, KeyError, UnicodeError) as exc:
                print(f"Skipped line {i}: {exc}", file=sys.stderr, flush=True)
            time.sleep(3)


if __name__ == "__main__":
    # Cross-process lock: protects shared train/eval/manifest from concurrent
    # collector instances, including orphaned jobs after dashboard restart.
    lock_dir = Path("training/dataset")
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / ".collection.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                "Já existe uma recolha em execução. "
                "Aguarda ou termina o processo anterior antes de iniciar outra."
            )
        main()
