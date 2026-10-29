"""Search adapter: model chooses queries; runtime fetches bounded search results safely."""
import json
import re
import urllib.parse
import urllib.request
from html import unescape

def search(query):
    query=str(query).strip()[:110]
    if len(query)<3: return []
    # Search-engine endpoint fixed; never fetch arbitrary links returned by results.
    url="https://www.google.com/search?"+urllib.parse.urlencode({"q":query,"num":"5"})
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 (compatible; NordAI/1.2)","Accept":"text/html"})
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs): raise ValueError("Redirect blocked")
    with urllib.request.build_opener(NoRedirect).open(req,timeout=8) as response:
        data=response.read(120001)
    if len(data)>120000:raise ValueError("Oversized search")
    # Google HTML is not a stable API; explicitly fail closed when parsing fails.
    html=data.decode("utf-8","replace")
    rows=[]
    for match in re.finditer(r'<a href="(https?[^"]+)"[^>]*>.*?<h3[^>]*>(.*?)</h3>',html,re.I|re.S):
        link=unescape(match.group(1))
        title=re.sub("<[^>]+>","",unescape(match.group(2))).strip()
        p=urllib.parse.urlsplit(link)
        if p.scheme!="https" or not p.hostname or p.hostname.endswith("google.com"):continue
        if len(title)<5:continue
        rows.append({"title":title[:140],"url":link[:450]})
        if len(rows)>=3:break
    return rows
