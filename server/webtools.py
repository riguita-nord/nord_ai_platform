"""Bounded web search. The model chooses queries and source IDs, not destinations."""
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html import unescape

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Redirecionamentos bloqueados")

def search(query):
    query=str(query).strip()[:110]
    if len(query)<3: return []
    url="https://www.bing.com/search?"+urllib.parse.urlencode({"q":query,"format":"rss","count":"5"})
    req=urllib.request.Request(url,headers={"User-Agent":"NordAI/1.2 (+private research assistant)","Accept":"application/rss+xml,application/xml"})
    with urllib.request.build_opener(NoRedirect).open(req,timeout=8) as response:
        body=response.read(100001)
    if len(body)>100000: raise ValueError("Pesquisa demasiado grande")
    root=ET.fromstring(body)
    results=[]
    for item in root.findall(".//channel/item")[:5]:
        title=(item.findtext("title") or "").strip()[:140]
        link=(item.findtext("link") or "").strip()
        description=re.sub(r"<[^>]+>"," ",unescape(item.findtext("description") or ""))
        description=re.sub(r"\s+"," ",description).strip()[:430]
        parsed=urllib.parse.urlsplit(link)
        if parsed.scheme!="https" or not parsed.hostname or len(title)<4: continue
        if parsed.username or parsed.password or parsed.port not in (None,443): continue
        results.append({"id":len(results)+1,"title":title,"url":link[:500],"snippet":description})
    return results
