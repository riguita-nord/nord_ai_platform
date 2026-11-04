"""Small, source-attributed public research for Nord Brain, CPU-only.

Search snippets are external, untrusted *evidence candidates*, NOT factual truth
or neural learning labels. Optional Brave API; keyless DuckDuckGo HTML fallback
is best-effort and unofficial. No arbitrary URL fetching or redirects.
"""
from __future__ import annotations

from html.parser import HTMLParser
import html
import json
import os
import re
import unicodedata
from urllib import parse, request

OFFICIAL = (
    (('python', 'programacao', 'linguagem python'),
     'https://docs.python.org/3/tutorial/index.html', 'Documentação oficial de Python'),
    (('raspberry', 'raspberry pi', 'computador pi'),
     'https://www.raspberrypi.com/documentation/computers/raspberry-pi.html', 'Documentação Raspberry Pi'),
    (('planeta', 'planetas', 'sistema solar', 'astronomia', 'nasa'),
     'https://science.nasa.gov/solar-system/planets/', 'NASA Science: planetas'),
    (('linux', 'kernel', 'nucleo linux'),
     'https://docs.kernel.org/admin-guide/index.html', 'Documentação do kernel Linux'),
)

# Editorial/primary sources can be useful for public facts, but their snippets
# are still not automatically verified or used as training labels.
TRUSTED_DOMAINS = (
    'britannica.com', 'reuters.com', 'apnews.com', 'bbc.com', 'nasa.gov',
    'nature.com', 'science.org', 'sciencemag.org', 'who.int', 'un.org',
    'europa.eu', 'gov.pt', 'raspberrypi.com', 'python.org',
    'docs.python.org', 'docs.kernel.org', 'spacex.com', 'tesla.com',
    'nobelprize.org', 'mit.edu', 'stanford.edu', 'harvard.edu',
    'nationalgeographic.com',
)


def normalize(text: str) -> str:
    lowered=unicodedata.normalize('NFKD',text.casefold())
    return ''.join(ch for ch in lowered if not unicodedata.combining(ch))


def source_for(topic: str):
    cleaned=normalize(topic.strip())
    if not 2<=len(cleaned)<=100:return None
    for words,url,title in OFFICIAL:
        if any(re.search(r'(?<!\w)'+re.escape(key)+r'(?!\w)',cleaned) for key in words):
            return url,title
    return None


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise ValueError('Redirecionamento bloqueado')


def _safe_query(topic: str) -> str | None:
    topic=' '.join(topic.strip().split())
    if not 2<=len(topic)<=100:return None
    # Do not leak personal information, tokens, URLs or credentials as a query.
    if re.search(r'(?i)https?://|www\.|\S+@\S+|(?:api[_ -]?key|password|senha|token|segredo|secret)\b',topic):
        return None
    if re.search(r'\b\d{9,}\b',topic):return None
    if re.search(r'[\x00-\x1f<>]',topic):return None
    return topic


def _read_response(url: str, *, headers: dict | None = None, max_bytes: int = 140000) -> bytes:
    req=request.Request(url,headers={'User-Agent':'NordBrain/1.2 (+source-attributed research)',
                                        'Accept':'text/html,application/json',**(headers or {})})
    with request.build_opener(NoRedirect).open(req,timeout=7) as response:
        if response.geturl().split('#')[0]!=url:
            raise ValueError('Resposta alterou o URL')
        return response.read(max_bytes+1)[:max_bytes]


def _canonical_url(url: str) -> str | None:
    if not url or len(url)>1000:return None
    if url.startswith('//'):url='https:'+url
    parsed=parse.urlsplit(url)
    if parsed.hostname in ('duckduckgo.com','www.duckduckgo.com'):
        qs=parse.parse_qs(parsed.query)
        if 'uddg' not in qs:return None
        url=qs['uddg'][0]
        parsed=parse.urlsplit(url)
    hostname=(parsed.hostname or '').lower()
    if (parsed.scheme!='https' or not hostname or parsed.username or parsed.password
            or hostname in ('localhost','127.0.0.1') or hostname.endswith('.local')
            or hostname.endswith('.internal') or re.fullmatch(r'[0-9.]+',hostname)):
        return None
    if parsed.port not in (None,443):return None
    if hostname.endswith(('wikipedia.org','wikimedia.org')):
        return None  # explicitly excluded in Nord Brain's chosen research policy
    return url.split('#')[0]


def _trusted(url: str) -> bool:
    host=(parse.urlsplit(url).hostname or '').lower()
    return any(host==domain or host.endswith('.'+domain) for domain in TRUSTED_DOMAINS)


def _text(value: str, limit: int = 450) -> str:
    value=html.unescape(re.sub(r'<[^>]+>',' ',str(value)))
    value=' '.join(value.split())
    return re.sub(r'[\x00-\x1f]+',' ',value).strip()[:limit]


def _to_results(rows,query,provider):
    seen=set(); result=[]
    wanted={word for word in normalize(query).split() if len(word)>2 and word not in ('quem','qual','onde','sobre','como','que','uma','dos','das')}
    for title,snippet,url in rows:
        canonical=_canonical_url(url)
        if not canonical or canonical in seen:continue
        title=_text(title,140);snippet=_text(snippet,390)
        if not title or len(snippet)<18:continue
        haystack=normalize(title+' '+snippet)
        if wanted and len(wanted.intersection(haystack.split()))<min(1,len(wanted)):continue
        seen.add(canonical)
        trusted=_trusted(canonical)
        result.append({'title':title,'summary':snippet,'url':canonical,
                       'source_tier':'trusted_discovery' if trusted else 'unverified_discovery',
                       'source_label':parse.urlsplit(canonical).hostname or '',
                       'provider':provider})
    result.sort(key=lambda item:(item['source_tier']=='trusted_discovery',
                                  sum(word in normalize(item['title']) for word in wanted)),reverse=True)
    return result[:4]


class _DDG(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows=[];self.active=None;self.buffer=[];self.href=''

    def handle_starttag(self,tag,attrs):
        options=dict(attrs)
        cls=set(options.get('class','').split())
        if tag=='a' and 'result__a' in cls:
            self.active='title';self.buffer=[];self.href=options.get('href','')
        elif tag in ('a','div','span') and (cls & {'result__snippet','result-snippet'}):
            self.active='snippet';self.buffer=[]

    def handle_data(self,data):
        if self.active:self.buffer.append(data)

    def handle_endtag(self,tag):
        if self.active=='title' and tag=='a':
            title=_text(''.join(self.buffer),140)
            if title:self.rows.append([title,'',self.href])
            self.active=None;self.buffer=[]
        elif self.active=='snippet' and tag in ('a','div','span'):
            if self.rows and not self.rows[-1][1]:self.rows[-1][1]=_text(''.join(self.buffer),390)
            self.active=None;self.buffer=[]


def _ddg_search(query: str):
    url='https://html.duckduckgo.com/html/?'+parse.urlencode({'q':query})
    data=_read_response(url)
    parser=_DDG();parser.feed(data.decode('utf-8','replace'))
    return [(title,snippet,link) for title,snippet,link in parser.rows]


def _brave_search(query: str,key: str):
    url='https://api.search.brave.com/res/v1/web/search?'+parse.urlencode({
        'q':query,'count':7,'safesearch':'strict','search_lang':'pt'
    })
    data=json.loads(_read_response(url,headers={'Accept':'application/json','X-Subscription-Token':key},max_bytes=180000))
    return [(x.get('title',''),x.get('description',''),x.get('url',''))
            for x in data.get('web',{}).get('results',[])[:7]]


class _SummaryParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title='';self.description='';self.paragraphs=[];self.capture=False;self.buff=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        opts=dict(attrs)
        if tag in ('script','style','nav','footer'):self.skip+=1
        if tag=='meta' and (opts.get('name','').casefold()=='description' or opts.get('property','').casefold()=='og:description'):
            self.description=self.description or opts.get('content','')
        if tag=='title' and not self.skip:self.capture='title';self.buff=[]
        if tag=='p' and not self.skip:self.capture='p';self.buff=[]
    def handle_data(self,data):
        if self.capture and not self.skip:self.buff.append(data)
    def handle_endtag(self,tag):
        if self.capture=='title' and tag=='title':self.title=_text(''.join(self.buff),140);self.capture=False
        elif self.capture=='p' and tag=='p':
            p=_text(''.join(self.buff),600)
            if len(p)>60:self.paragraphs.append(p)
            self.capture=False
        if tag in ('script','style','nav','footer'):self.skip=max(0,self.skip-1)


# Legacy name retained for the source-extraction regression suite.
_Extract = _SummaryParser


def _fetch_fixed(url: str) -> tuple[str,str]:
    if url not in {s[1] for s in OFFICIAL}:raise ValueError('Endereço não autorizado')
    data=_read_response(url,max_bytes=120000)
    parser=_SummaryParser();parser.feed(data.decode('utf-8','replace'))
    title=parser.title
    summary=_text(parser.description or next(iter(parser.paragraphs),''),400)
    if not 35<=len(summary)<=400:raise ValueError('Sem resumo útil')
    return title,summary[:220]


def search(topic: str):
    """Public-source discovery. An empty result means the search was inconclusive.

    Does not silently trust snippets or update model parameters. User approves
    a source in the conversation before any text is used as a training label.
    """
    query=_safe_query(topic)
    if query is None:return []
    selected=source_for(query)
    if selected:
        url,label=selected
        try:
            title,summary=_fetch_fixed(url)
            return [{'title':title or label,'summary':summary,'url':url,
                     'source_tier':'official_primary','source_label':label,'provider':'fixed'}]
        except (OSError,ValueError,TimeoutError):
            pass
    key=os.getenv('NORD_BRAVE_API_KEY','').strip()
    try:
        if key:
            rows=_brave_search(query,key)
            results=_to_results(rows,query,'brave')
            if results:return results
    except (OSError,ValueError,KeyError,TimeoutError,TypeError,json.JSONDecodeError):
        pass
    try:
        return _to_results(_ddg_search(query),query,'duckduckgo-html')
    except (OSError,ValueError,TimeoutError):
        return []


def public_question(message: str) -> str | None:
    """Extract a public research topic, without interpreting it as a fact.

    This is an intent guard, not a free-form Portuguese language model.
    """
    text=' '.join(message.strip().split()).strip('?! .')
    m=re.fullmatch(r'(?:e\s+)?(?:diz(?:-me| me)?\s+)?(?:quem\s+(?:é|e|foi)\s+(?:o|a)?\s*|o\s+que\s+(?:é|e|significa)\s+|o\s+que\s+sabes\s+sobre\s+|onde\s+fica\s+|quando\s+(?:foi|é|e)\s+|fala(?:-me| me)?\s+sobre\s+)(.+)',text,re.I)
    if m:return _safe_query(m[1])
    return None
