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
import urllib.parse

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
    try:
        webtools._public_https(url)
    except Exception as exc:
        raise RuntimeError("Endereço/rede bloqueados: "+str(exc)[:140]) from exc
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
            raise RuntimeError("Chromium terminou com código "+str(process.returncode)+" (verificar sandbox, Snap e memória)")
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
        raise RuntimeError("Página sem texto legível; pode exigir cookies, login ou outro tipo de conteúdo")
    return extracted[:3600]

class GoogleResults(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links=[]
        self.href=None
        self.in_h3=0
        self.label=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="a":
            self.href=attrs.get("href")
        if tag=="h3" and self.href:
            self.in_h3+=1
            self.label=[]
    def handle_data(self,data):
        if self.in_h3:self.label.append(data)
    def handle_endtag(self,tag):
        if tag=="h3" and self.in_h3:
            self.in_h3-=1
            title=" ".join(self.label).strip()
            if title and self.href:
                self.links.append((title,self.href))
        if tag=="a":
            self.href=None

def google_search(query,on_snapshot=None):
    """Open Google Search in Chromium, parse ranked result links from rendered DOM."""
    query=str(query).strip()[:130]
    if len(query)<3:return []
    url="https://www.google.com/search?"+urllib.parse.urlencode({"q":query,"num":10,"hl":"pt"})
    # Separate bounded render for extracting links; Chromium is terminated on completion.
    webtools._public_https(url)
    binary=browser_path()
    with tempfile.TemporaryDirectory(prefix="nord-google-") as profile:
        command=[binary,"--headless=new","--disable-gpu","--disable-dev-shm-usage",
                 "--disable-extensions","--disable-background-networking","--no-first-run",
                 "--no-default-browser-check","--window-size=1280,900",
                 "--virtual-time-budget=3000","--dump-dom",
                 "--user-data-dir="+profile,url]
        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL,start_new_session=True)
        try:
            data,_=proc.communicate(timeout=22)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.communicate()
            raise RuntimeError("Pesquisa Google excedeu 22 segundos")
        if proc.returncode or len(data)>1_500_000:
            raise RuntimeError("Chromium não conseguiu obter resultados Google")
    parsed=GoogleResults()
    parsed.feed(data.decode("utf-8","replace"))
    results=[]
    seen=set()
    for title,href in parsed.links:
        target=urllib.parse.urljoin(url,unescape(href))
        bits=urllib.parse.urlsplit(target)
        if bits.hostname in ("www.google.com","google.com"):
            param=urllib.parse.parse_qs(bits.query)
            target=(param.get("q") or param.get("url") or [""])[0]
        try:
            webtools._public_https(target)
        except Exception:continue
        bits=urllib.parse.urlsplit(target)
        canonical=(bits.hostname or "").lower()+bits.path.rstrip("/")
        if canonical in seen or not canonical:continue
        seen.add(canonical)
        results.append({"id":len(results)+1,"title":title[:160],"url":target[:650],
                        "snippet":"Resultado apresentado pelo Google"})
        if len(results)==6:break
    if on_snapshot:
        # Screenshot is observational only; it does not grant interactive control.
        try:render({"url":url},on_snapshot=on_snapshot)
        except Exception:pass
    return results
