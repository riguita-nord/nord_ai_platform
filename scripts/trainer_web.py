"""Nord AI Trainer: local authenticated training dashboard (Python stdlib).

Start with: .venv/bin/python scripts/trainer_web.py
Default bind is 127.0.0.1. For LAN use --host 0.0.0.0 --token SECRET
and a trusted private network; HTTP is not encrypted.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rebuild_language.py"
PAGE = ROOT / "scripts" / "trainer_ui.html"
DATASET = ROOT / "training" / "dataset" / "train.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.environ.get("NORD_TRAINER_TOKEN"))
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".nord-brain")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        parser.error("LAN binding requires --token or NORD_TRAINER_TOKEN")
    token = args.token or secrets.token_urlsafe(24)
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "trainer.log"
    state = {"process": None, "last_exit": None}
    guard = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def authorized(self):
            supplied = self.headers.get("X-Nord-Token", "")
            return hmac.compare_digest(supplied, token)

        def send_json(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                html = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy",
                                 "default-src 'none'; style-src 'unsafe-inline'; "
                                 "script-src 'unsafe-inline'; connect-src 'self'; "
                                 "base-uri 'none'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(html)
                return
            if not self.authorized():
                self.send_json(401, {"error": "Token inválido"})
                return
            if path == "/api/status":
                with guard:
                    process = state["process"]
                    running = process is not None and process.poll() is None
                    code = process.poll() if process is not None else state["last_exit"]
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-70:] if log_path.exists() else []
                self.send_json(200, {"running": running, "exit_code": code,
                                     "log": "\n".join(lines),
                                     "dataset_bytes": DATASET.stat().st_size if DATASET.exists() else 0})
                return
            self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if not self.authorized():
                self.send_json(401, {"error": "Token inválido"})
                return
            path = urlsplit(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 8192:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length)) if length else {}
                if not isinstance(payload, dict):
                    raise ValueError("Invalid request")
                if path == "/api/start":
                    steps = int(payload.get("steps", 1000))
                    ratio = float(payload.get("text_ratio", .75))
                    reset = payload.get("reset", False)
                    if not 1 <= steps <= 100000 or not .1 <= ratio <= .95 or type(reset) is not bool:
                        raise ValueError("Invalid training configuration")
                    if not DATASET.is_file() or DATASET.stat().st_size < 1000000:
                        raise ValueError("Dataset de treino precisa de pelo menos 1 MB")
                    with guard:
                        process = state["process"]
                        if process is not None and process.poll() is None:
                            self.send_json(409, {"error": "Já existe um treino em execução"})
                            return
                        cmd = [sys.executable, "-u", str(SCRIPT), "--corpus", str(DATASET),
                               "--data-dir", str(data_dir), "--steps", str(steps),
                               "--text-ratio", str(ratio), "--save-every", "50", "--preview"]
                        if reset:
                            cmd.append("--reset")
                        with log_path.open("w", encoding="utf-8") as logfile:
                            state["process"] = subprocess.Popen(
                                cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=logfile, stderr=subprocess.STDOUT,
                                start_new_session=True)
                            state["last_exit"] = None
                    self.send_json(202, {"started": True})
                    return
                if path == "/api/stop":
                    with guard:
                        process = state["process"]
                        if process is None or process.poll() is not None:
                            self.send_json(409, {"error": "Não existe treino ativo"})
                            return
                        # SIGINT allows the trainer to finish its current step.
                        # The last saved checkpoint remains durable.
                        process.send_signal(signal.SIGINT)
                    self.send_json(202, {"stopping": True})
                    return
                self.send_json(404, {"error": "Not found"})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except OSError as exc:
                self.send_json(500, {"error": str(exc)})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Nord Trainer: http://{args.host}:{args.port}", flush=True)
    print(f"Access token: {token}", flush=True)
    print("Stop ./nord before starting a training session. Ctrl+C stops the dashboard.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
