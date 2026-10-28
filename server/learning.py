"""Nord Learning Center: bounded search of a fixed public source + approved SQLite FTS5."""
import json
import re
import threading
import time
import urllib.parse
import urllib.request

TOPICS=("Programação","Inteligência artificial","Tecnologia","Ciência","Escrita criativa","FiveM / QBCore")
MAX_EXCERPT=2800
MAX_ITEMS=800
search_lock=threading.BoundedSemaphore(1)
HEADERS={"User-Agent":"NordAI-LearningCenter/1.1 (private educational client)","Accept":"application/json"}

def init(db):
    db.execute("""CREATE TABLE IF NOT EXISTS learn_items (
      id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, title TEXT NOT NULL,
      url TEXT NOT NULL UNIQUE, excerpt TEXT NOT NULL, status TEXT NOT NULL,
      created INTEGER NOT NULL)""")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS learn_fts USING fts5(title,excerpt,content='learn_items',content_rowid='id')")
    db.executescript("""CREATE TRIGGER IF NOT EXISTS learn_ai AFTER INSERT ON learn_items BEGIN
      INSERT INTO learn_fts(rowid,title,excerpt) VALUES(new.id,new.title,new.excerpt); END;
      CREATE TRIGGER IF NOT EXISTS learn_ad AFTER DELETE ON learn_items BEGIN
      INSERT INTO learn_fts(learn_fts,rowid,title,excerpt) VALUES('delete',old.id,old.title,old.excerpt); END;
      CREATE TRIGGER IF NOT EXISTS learn_au AFTER UPDATE ON learn_items BEGIN
      INSERT INTO learn_fts(learn_fts,rowid,title,excerpt) VALUES('delete',old.id,old.title,old.excerpt);
      INSERT INTO learn_fts(rowid,title,excerpt) VALUES(new.id,new.title,new.excerpt); END;""")
    db.commit()

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError("Redirecionamento externo não permitido")

def discover(query,topic):
    if topic not in TOPICS: raise ValueError("Área inválida")
    query=str(query).strip()
    if len(query)<2 or len(query)>100: raise ValueError("A pesquisa deve ter entre 2 e 100 caracteres")
    if not search_lock.acquire(blocking=False): raise ValueError("Já existe uma pesquisa em curso")
    try:
        # No arbitrary user URLs: fixed HTTPS endpoint; redirects refused.
        args=urllib.parse.urlencode({
          "action":"query","generator":"search","gsrsearch":query,"gsrlimit":"5",
          "prop":"extracts","explaintext":"1","exintro":"1","exsentences":"7",
          "format":"json","formatversion":"2"})
        request=urllib.request.Request("https://pt.wikipedia.org/w/api.php?"+args,headers=HEADERS)
        with urllib.request.build_opener(NoRedirect).open(request,timeout=8) as response:
            data=response.read(100001)
        if len(data)>100000: raise ValueError("Resposta externa demasiado grande")
        result=json.loads(data)
        found=[]
        for item in result.get("query",{}).get("pages",[]):
            pid=item.get("pageid")
            if not isinstance(pid,int) or pid<=0: continue
            title=str(item.get("title","")).strip()[:180]
            excerpt=re.sub(r"\s+"," ",str(item.get("extract",""))).strip()[:MAX_EXCERPT]
            if not title or len(excerpt)<50:continue
            found.append({"topic":topic,"title":title,"excerpt":excerpt,
                          "url":"https://pt.wikipedia.org/?curid="+str(pid)})
        return found
    finally:
        search_lock.release()

def _valid_item(item):
    if item.get("topic") not in TOPICS:raise ValueError("Área inválida")
    url=str(item.get("url",""))
    parsed=urllib.parse.urlparse(url)
    qs=urllib.parse.parse_qs(parsed.query)
    if (parsed.scheme!="https" or parsed.netloc!="pt.wikipedia.org" or parsed.path!="/"
        or parsed.fragment or set(qs)!={"curid"} or len(qs["curid"])!=1
        or not re.fullmatch(r"[0-9]{1,12}",qs["curid"][0])):
        raise ValueError("URL da fonte não permitida")
    title=str(item.get("title","")).strip()[:180]
    excerpt=str(item.get("excerpt","")).strip()[:MAX_EXCERPT]
    if not title or len(excerpt)<50:raise ValueError("Conteúdo insuficiente")
    return item["topic"],title,url,excerpt

def save_discovery(db,results):
    added=0
    for item in results[:5]:
        topic,title,url,excerpt=_valid_item(item)
        if db.execute("SELECT COUNT(*) FROM learn_items").fetchone()[0]>=MAX_ITEMS:break
        cur=db.execute("INSERT OR IGNORE INTO learn_items(topic,title,url,excerpt,status,created) VALUES(?,?,?,?,?,?)",
                       (topic,title,url,excerpt,"approved",int(time.time())))
        added+=cur.rowcount
    db.commit()
    return added

def listing(db,topic=None):
    if topic and topic not in TOPICS:raise ValueError("Área inválida")
    if topic:
        rows=db.execute("SELECT id,topic,title,url,excerpt,created FROM learn_items WHERE topic=? ORDER BY id DESC LIMIT 100",(topic,)).fetchall()
    else:
        rows=db.execute("SELECT id,topic,title,url,excerpt,created FROM learn_items ORDER BY id DESC LIMIT 100").fetchall()
    return [dict(zip(("id","topic","title","url","excerpt","created"),r)) for r in rows]

def delete(db,item_id):
    cur=db.execute("DELETE FROM learn_items WHERE id=?",(int(item_id),))
    db.commit()
    if not cur.rowcount:raise ValueError("Registo não encontrado")

def recall(db,question):
    terms=re.findall(r"[^\W_]{3,}",question.lower(),re.UNICODE)[:6]
    if not terms:return []
    query=" OR ".join('"'+term+'"' for term in terms)
    try:
        rows=db.execute("""SELECT i.title,i.url,i.excerpt FROM learn_fts
            JOIN learn_items i ON i.id=learn_fts.rowid
            WHERE learn_fts MATCH ? AND i.status='approved'
            ORDER BY bm25(learn_fts) LIMIT 2""",(query,)).fetchall()
    except Exception:
        return []
    return [{"title":title,"url":url,"excerpt":excerpt[:850]} for title,url,excerpt in rows]
