"""Opt-in, size-bounded HTTPS corpus import for offline Nord training."""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
import socket
import urllib.parse
import urllib.request

MAX_BYTES=95000

def import_text(url: str, directory: Path) -> tuple[Path,dict]:
    parsed=urllib.parse.urlsplit(url)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('A fonte deve ser um URL HTTPS público')
    if parsed.port not in (None,443):
        raise ValueError('Porta HTTPS não suportada')
    for family,_,_,_,address in socket.getaddrinfo(parsed.hostname,443,type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(address[0]).is_global:
            raise ValueError('A fonte não pode apontar para uma rede privada')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):
            raise ValueError('Redirecionamentos não são permitidos')
    opener=urllib.request.build_opener(NoRedirect)
    request=urllib.request.Request(url,headers={'User-Agent':'NordBrain-Corpus/1.0','Accept':'text/plain'})
    with opener.open(request,timeout=12) as response:
        mime=response.headers.get_content_type()
        if mime!='text/plain':
            raise ValueError('Apenas fontes text/plain são aceites')
        if int(response.headers.get('Content-Length') or 0)>MAX_BYTES:
            raise ValueError('Fonte demasiado grande')
        raw=response.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:
        raise ValueError('Fonte demasiado grande')
    try:
        content=raw.decode('utf-8')
    except UnicodeError as exc:
        raise ValueError('A fonte deve estar em UTF-8') from exc
    if len(content.strip())<180:
        raise ValueError('Fonte demasiado curta')
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    import hashlib
    digest=hashlib.sha256(raw).hexdigest()
    target=directory/('source-'+digest[:16]+'.txt')
    target.write_text(content,encoding='utf-8')
    metadata={'url':url,'sha256':digest,'bytes':len(raw),'approved':False}
    target.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return target,metadata
