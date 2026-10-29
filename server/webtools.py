"""Conservative web research tools for the Nord AI runtime.

Search queries are proposed by the model or derived from explicit user intent.
Web pages can only be fetched from links returned by the search provider.
"""
import ipaddress
import re
import socket
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser

MAX_SEARCH_BYTES=100_000
MAX_PAGE_BYTES=450_000

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError("Redirecionamento bloqueado")

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts=[]
        self.ignore=0
    def handle_starttag(self,tag,attrs):
        if tag in ("script","style","noscript","svg","iframe","form"): self.ignore+=1
    def handle_endtag(self,tag):
        if tag in ("script","style","noscript","svg","iframe","form"):
            self.ignore=max(0,self.ignore-1)
        if tag in ("p","h1","h2","h3","li"):self.parts.append(" ")
    def handle_data(self,data):
        if not self.ignore:self.parts.append(data)

def _public_https(url):
    parsed=urllib.parse.urlsplit(url)
    if parsed.scheme!="https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL inválido")
    try:
        if parsed.port not in (None,443):raise ValueError("Porta inválida")
    except ValueError:
        raise ValueError("Porta inválida")
    host=parsed.hostname.lower().rstrip(".")
    if host=="localhost" or host.endswith((".local",".internal",".localhost")):
        raise ValueError("Destino local bloqueado")
    # DNS names resolve only to globally routable IPs. DNS rebinding remains a risk:
    # production deployment should also enforce egress network firewall restrictions.
    addresses=socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)
    if not addresses:raise ValueError("DNS sem resultados")
    for item in addresses:
        ip=ipaddress.ip_address(item[4][0].split("%",1)[0])
        if not ip.is_global:raise ValueError("IP privado ou reservado bloqueado")
    return parsed

def _read(url,limit,accept):
    _public_https(url)
    req=urllib.request.Request(url,headers={
        "User-Agent":"NordAI/1.3 (small private research assistant)",
        "Accept":accept})
    with urllib.request.build_opener(NoRedirect).open(req,timeout=7) as resp:
        final=resp.geturl()
        if final!=url:raise ValueError("Destino alterado")
        data=resp.read(limit+1)
        content_type=resp.headers.get("Content-Type","").lower()
    if len(data)>limit:raise ValueError("Documento demasiado grande")
    return data,content_type

def search(query):
    query=str(query).strip()[:110]
    if len(query)<3:return []
    url="https://www.bing.com/search?"+urllib.parse.urlencode({"q":query,"format":"rss","count":"5"})
    data,content_type=_read(url,MAX_SEARCH_BYTES,"application/rss+xml,application/xml")
    root=ET.fromstring(data)
    results=[]
    for item in root.findall(".//channel/item")[:5]:
        title=(item.findtext("title") or "").strip()[:160]
        link=(item.findtext("link") or "").strip()
        snippet=re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",unescape(item.findtext("description") or ""))).strip()[:550]
        try:_public_https(link)
        except Exception:continue
        if title and snippet:
            results.append({"id":len(results)+1,"title":title,"url":link[:700],"snippet":snippet})
    return results

def read_page(source):
    """Read one search-returned URL, stripping scripts and forms."""
    url=source["url"]
    data,content_type=_read(url,MAX_PAGE_BYTES,"text/html,text/plain")
    if not any(token in content_type for token in ("text/html","text/plain")):
        raise ValueError("Formato de página não suportado")
    text=data.decode("utf-8","replace")
    if "text/html" in content_type:
        parser=TextExtractor()
        parser.feed(text)
        text=" ".join(parser.parts)
    text=re.sub(r"\s+"," ",unescape(text)).strip()
    return text[:3600]
