"""Headless Chromium reader with bounded lifetime, output and temporary profile."""
import os
import re
import shutil
import signal
import subprocess
import tempfile
import resource
import base64
from pathlib import Path
from html import unescape
from html.parser import HTMLParser
import webtools

class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts=[]
        self.hidden=0
    def handle_starttag(self,tag,attrs):
        if tag in ("script","style","noscript","svg","iframe","form","template"):
            self.hidden+=1
    def handle_endtag(self,tag):
        if tag in ("script","style","noscript","svg","iframe","form","template"):
            self.hidden=max(0,self.hidden-1)
        if tag in ("p","h1","h2","h3","h4","li","article","section","tr"):
            self.parts.append(" ")
    def handle_data(self,data):
        if not self.hidden:
            self.parts.append(data)

def browser_path():
    for name in ("chromium","chromium-browser"):
        cmd=shutil.which(name)
        if cmd:return cmd
    raise RuntimeError("Chromium não está instalado. Executa sudo bash install.sh.")

def render(source, on_snapshot=None):
    url=source["url"]
    webtools._public_https(url)
    binary=browser_path()
    with tempfile.TemporaryDirectory(prefix="nord-chromium-") as profile:
        cmd=[
            binary,"--headless=new","--disable-gpu","--disable-dev-shm-usage",
            "--disable-extensions","--disable-background-networking","--disable-sync",
            "--no-first-run","--no-default-browser-check",
            "--no-crash-upload","--disable-component-update",
            "--hide-scrollbars","--window-size=1280,900",
            "--virtual-time-budget=3500","--dump-dom",
            "--user-data-dir="+profile,url
        ]
        # Chromium runs as the unprivileged nordai service user, with its sandbox
        # enabled. Never add --no-sandbox to make it launch.
        process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL,start_new_session=True)
        try:
            # Do not retain a complete arbitrarily large rendered page.
            out,_=process.communicate(timeout=22)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.communicate()
            raise RuntimeError("O Chromium ultrapassou o limite de 22 segundos")
        if process.returncode:
            raise RuntimeError("O Chromium não conseguiu renderizar a página (sandbox ou memória)")
        if len(out)>1_500_000:
            raise RuntimeError("Página demasiado grande para o browser interno")
    # A small read-only snapshot, captured with a separate bounded Chromium run.
    # No browser control surface, cookies, persistent profile or debug port is exposed.
    if on_snapshot is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="nord-preview-") as preview_dir:
                screenshot=Path(preview_dir)/"page.png"
                cmd=[binary,"--headless=new","--disable-gpu","--disable-dev-shm-usage",
                     "--disable-extensions","--disable-background-networking","--no-first-run",
                     "--no-default-browser-check","--hide-scrollbars","--window-size=960,650",
                     "--virtual-time-budget=1600","--user-data-dir="+str(Path(preview_dir)/"profile"),
                     "--screenshot="+str(screenshot),url]
                proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL,start_new_session=True)
                try:proc.wait(timeout=14)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
                if proc.returncode==0 and screenshot.exists() and screenshot.stat().st_size<=350000:
                    on_snapshot(base64.b64encode(screenshot.read_bytes()).decode("ascii"))
        except Exception:
            pass
    parser=VisibleText()
    parser.feed(out.decode("utf-8","replace"))
    extracted=re.sub(r"\s+"," ",unescape(" ".join(parser.parts))).strip()
    if len(extracted)<80:
        raise RuntimeError("A página renderizada não contém texto suficiente")
    return extracted[:3600]
