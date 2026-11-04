"""Bounded research restricted to maintained primary-source documentation.

No Wikipedia, arbitrary URLs, redirects, browser execution or free-text web
search. Topics map to *constant* HTTPS pages; personal chat text is never sent
as a query string. Content is source-attributed but NOT automatically treated
as proof of a fact. Site markup is data, not instructions.
"""
from __future__ import annotations

from html.parser import HTMLParser
import re
import unicodedata
from urllib import request, parse


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


def normalize(text: str):
    lowered=unicodedata.normalize('NFKD',text.casefold())
    return ''.join(ch for ch in lowered if not unicodedata.combining(ch))


def source_for(topic: str):
    cleaned=normalize(topic.strip())
    if not 2 <= len(cleaned) <= 100:
        return None
    # Never forward open-ended chat/questions to a remote search engine.
    for words,url,title in OFFICIAL:
        if any(re.search(r'(?<!\w)'+re.escape(key)+r'(?!\w)',cleaned) for key in words):
            return url,title
    return None


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Redirecionamento bloqueado')


class _Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title=[]
        self.paragraphs=[]
        self.description=''
        self._in_title=False
        self._in_para=False
        self._para=[]
        self._skip=0

    def handle_starttag(self, tag, attrs):
        options=dict(attrs)
        if tag in ('script','style','nav','footer'):
            self._skip+=1
        if tag=='title':self._in_title=True
        if tag=='meta' and options.get('name','').lower() in ('description','og:description'):
            self.description=self.description or options.get('content','')
        if tag=='meta' and options.get('property','').lower()=='og:description':
            self.description=self.description or options.get('content','')
        if tag=='p' and not self._skip:
            self._in_para=True
            self._para=[]

    def handle_endtag(self, tag):
        if tag=='title':self._in_title=False
        if tag=='p' and self._in_para:
            paragraph=' '.join(''.join(self._para).split())
            if 35 <= len(paragraph) <= 1200:self.paragraphs.append(paragraph)
            self._in_para=False
        if tag in ('script','style','nav','footer'):
            self._skip=max(0,self._skip-1)

    def handle_data(self, text):
        if self._skip:return
        if self._in_title:self.title.append(text)
        if self._in_para:self._para.append(text)


def _fetch_fixed(url: str) -> tuple[str,str]:
    if url not in {s[1] for s in OFFICIAL}:
        raise ValueError('Endereço não autorizado')
    req=request.Request(url, headers={
        'User-Agent':'NordBrain/0.9 (research prototype; rate-limited; source-attributed)',
        'Accept':'text/html',
    })
    opener=request.build_opener(NoRedirect)
    with opener.open(req,timeout=6) as response:
        if response.geturl().split('#')[0]!=url:
            raise ValueError('Origem de resposta diferente da autorizada')
        ctype=response.headers.get('Content-Type','text/html').lower()
        if 'text/html' not in ctype:
            raise ValueError('Conteúdo não HTML')
        # Read only the beginning: large official pages may have long HTML,
        # but descriptions and introductions live near the start. Never fetch
        # megabytes of content into a Raspberry Pi process.
        raw=response.read(120000)
    parser=_Extract()
    parser.feed(raw.decode('utf-8',errors='replace'))
    title=' '.join(''.join(parser.title).split())[:120]
    summary=' '.join((parser.description or next(iter(parser.paragraphs), '')).split())
    # Fail closed: boilerplate and suspicious instruction patterns are not
    # admitted as learning examples.
    summary=re.sub(r'[\x00-\x1f]+',' ',summary).strip()
    if not 35 <= len(summary) <= 2000:
        raise ValueError('A fonte não tem resumo utilizável')
    if re.search(r'(?i)ignore (all |the )?previous|system prompt|execute (this |the )?command|api.?key|password',summary):
        raise ValueError('Conteúdo com instruções não autorizado')
    return title,summary[:220]


def search(topic: str):
    selected=source_for(topic)
    if not selected:return []
    url,label=selected
    title,summary=_fetch_fixed(url)
    return [{'title':title or label,'summary':summary,'url':url,
             'source_tier':'official_primary', 'source_label':label}]
