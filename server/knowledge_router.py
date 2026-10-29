"""Knowledge router: bounded durable cache with sources and freshness windows."""
import hashlib
import json
import re
import time

MAX_RECORDS=250

def normalize(text):
    text=re.sub(r"https?://\\S+"," ",str(text).lower())
    words=re.findall(r"[^\\W_]{3,}",text,re.UNICODE)
    stop={"como","funciona","quais","qual","para","sobre","uma","uns","com","que","isso","mais","explica","podes","fazer","quero","porque","este","esta","estão","tens","seria"}
    return " ".join(w for w in words if w not in stop)[:240]

def init(db):
    db.execute("""CREATE TABLE IF NOT EXISTS research_cache(
      key TEXT PRIMARY KEY, question TEXT NOT NULL, topic TEXT NOT NULL,
      evidence TEXT NOT NULL, sources TEXT NOT NULL,
      retrieved INTEGER NOT NULL, expires INTEGER NOT NULL)""")
    db.execute("CREATE INDEX IF NOT EXISTS research_cache_expires ON research_cache(expires)")
    db.commit()

def dynamic(question):
    return bool(re.search(r"\\b(hoje|agora|atualmente|recente\\w*|últim\\w*|preço\\w*|lançamento\\w*|notícia\\w*|versã\\w*|202[4-9])\\b",question.lower()))

def cache_key(question):
    return hashlib.sha256(normalize(question).encode()).hexdigest()

def recall(db,question):
    now=int(time.time())
    key=cache_key(question)
    row=db.execute("SELECT evidence,sources,retrieved,expires FROM research_cache WHERE key=? AND expires>?",(key,now)).fetchone()
    if not row:return None
    return {"evidence":row[0],"sources":json.loads(row[1]),"retrieved":row[2],"expires":row[3]}

def save(db,question,evidence,sources):
    if not sources or not evidence.strip():return
    now=int(time.time())
    ttl=3600 if dynamic(question) else 14*24*3600
    db.execute("INSERT OR REPLACE INTO research_cache VALUES(?,?,?,?,?,?,?)",(
        cache_key(question),question[:500],normalize(question),evidence[:2800],
        json.dumps(sources[:3],ensure_ascii=False),now,now+ttl))
    db.execute("DELETE FROM research_cache WHERE expires<?",(now,))
    db.execute("DELETE FROM research_cache WHERE key IN (SELECT key FROM research_cache ORDER BY retrieved DESC LIMIT -1 OFFSET ?)",(MAX_RECORDS,))
    db.commit()

def route_hint(prompt):
    q=prompt.lower().strip()
    if re.search(r"https?://",q):return "link"
    if re.search(r"(?:pesquis\\w*|investig\\w*|procur\\w*|consult\\w*)\\s+(?:na\\s+)?(?:internet|web|online)",q):return "web"
    if re.search(r"\\b(hoje|últimas? notícias|preços? atuais|agora|atualmente)\\b",q):return "web"
    if re.search(r"^(?:olá|ola|bom dia|boa tarde|obrigad\\w*|resume|traduz|escreve|inventa|cria uma história)",q):return "local"
    return "decide"
