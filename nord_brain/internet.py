"""Optional, bounded, source-aware Wikipedia reader; never trains by itself."""
import json
import re
from urllib import parse,request


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Redirecionamentos proibidos')


def search(topic: str):
    topic=topic.strip()
    if not 2 <= len(topic) <= 90:
        raise ValueError('Pesquisa entre 2 e 90 caracteres')
    query=parse.urlencode({'action':'query','generator':'search','gsrsearch':topic,
        'gsrlimit':3,'prop':'extracts','exintro':1,'explaintext':1,
        'exsentences':5,'format':'json','formatversion':2})
    req=request.Request('https://pt.wikipedia.org/w/api.php?'+query,
        headers={'User-Agent':'NordBrain/0.1 (educational terminal prototype)', 'Accept':'application/json'})
    with request.build_opener(NoRedirect).open(req,timeout=7) as response:
        if response.url.split('?',1)[0] != 'https://pt.wikipedia.org/w/api.php':
            raise ValueError('Origem inesperada')
        data=response.read(60001)
    if len(data)>60000:
        raise ValueError('Resposta demasiado grande')
    pages=json.loads(data).get('query',{}).get('pages',[])
    results=[]
    for p in pages:
        pid=p.get('pageid')
        if not isinstance(pid,int) or pid<=0: continue
        excerpt=re.sub(r'\s+',' ',str(p.get('extract',''))).strip()[:1500]
        if len(excerpt)<20: continue
        results.append({'title':str(p.get('title',''))[:120],
                        'summary':excerpt,'url':f'https://pt.wikipedia.org/?curid={pid}'})
    return results
