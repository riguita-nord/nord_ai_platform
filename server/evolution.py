"""Nord Evolution Engine: reviewed corrections, deterministic eval metadata, model version registry."""
import hashlib
import json
import re
import secrets
import time

BASE="gemma3:270m"
MAX_EXAMPLES=3000

def init(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS evolution_examples(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      prompt TEXT NOT NULL, correction TEXT NOT NULL, source TEXT NOT NULL,
      created INTEGER NOT NULL, approved INTEGER NOT NULL DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS evolution_versions(
      id TEXT PRIMARY KEY, model TEXT UNIQUE NOT NULL, parent TEXT,
      dataset_sha TEXT NOT NULL, eval_score REAL NOT NULL,
      baseline_score REAL NOT NULL, status TEXT NOT NULL, created INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS evolution_config(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    db.execute("INSERT OR IGNORE INTO evolution_config VALUES('active_model',?)",(BASE,))
    db.commit()

def add(db,prompt,correction):
    prompt=str(prompt).strip()
    correction=str(correction).strip()
    if not (3<=len(prompt)<=1800 and 3<=len(correction)<=2400):
        raise ValueError("Pergunta e correção devem ter entre 3 e 1800/2400 caracteres")
    if db.execute("SELECT COUNT(*) FROM evolution_examples").fetchone()[0]>=MAX_EXAMPLES:
        raise ValueError("Limite de exemplos atingido")
    db.execute("INSERT INTO evolution_examples(prompt,correction,source,created) VALUES(?,?,?,?)",
               (prompt,correction,"user_correction",int(time.time())))
    db.commit()

def dataset(db):
    rows=db.execute("SELECT id,prompt,correction FROM evolution_examples WHERE approved=1 ORDER BY id").fetchall()
    return [{"id":i,"messages":[{"role":"user","content":p},{"role":"assistant","content":c}]} for i,p,c in rows]

def digest(items):
    return hashlib.sha256(json.dumps(items,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def overview(db):
    active=db.execute("SELECT value FROM evolution_config WHERE key='active_model'").fetchone()[0]
    rows=db.execute("SELECT id,model,parent,dataset_sha,eval_score,baseline_score,status,created FROM evolution_versions ORDER BY created DESC LIMIT 30").fetchall()
    return {"active":active,"base":BASE,
      "examples":db.execute("SELECT COUNT(*) FROM evolution_examples WHERE approved=1").fetchone()[0],
      "versions":[dict(zip(("id","model","parent","dataset_sha","eval_score","baseline_score","status","created"),row)) for row in rows]}

def register(db,model,parent,dataset_sha,score,baseline):
    model=str(model)
    if not re.fullmatch(r"nord-evo:[a-z0-9][a-z0-9._-]{0,48}",model):
        raise ValueError("Nome inválido: usa nord-evo:versao")
    if parent!=overview(db)["active"]:
        raise ValueError("O modelo de origem não é o ativo")
    if dataset_sha!=digest(dataset(db)):raise ValueError("Dataset mudou desde o treino")
    score=float(score);baseline=float(baseline)
    if not (0<=score<=1 and 0<=baseline<=1):raise ValueError("Scores inválidos")
    if score<=baseline:raise ValueError("Avaliação não melhorou o baseline")
    vid=secrets.token_hex(8)
    db.execute("INSERT INTO evolution_versions VALUES(?,?,?,?,?,?,?,?)",
               (vid,model,parent,dataset_sha,score,baseline,"candidate",int(time.time())))
    db.commit()
    return vid

def switch(db,model,installed):
    if model!=BASE:
        row=db.execute("SELECT id FROM evolution_versions WHERE model=?",(model,)).fetchone()
        if not row:raise ValueError("Versão não registada")
    if model not in installed:raise ValueError("Modelo não instalado no Ollama")
    old=overview(db)["active"]
    db.execute("UPDATE evolution_versions SET status='inactive' WHERE status='active'")
    db.execute("UPDATE evolution_versions SET status='active' WHERE model=?",(model,))
    db.execute("UPDATE evolution_config SET value=? WHERE key='active_model'",(model,))
    db.commit()
    return old

def examples(db):
    return [{"id":x[0],"prompt":x[1],"correction":x[2],"created":x[3]} for x in db.execute(
      "SELECT id,prompt,correction,created FROM evolution_examples ORDER BY id DESC LIMIT 50")]

def remove(db,id):
    cur=db.execute("DELETE FROM evolution_examples WHERE id=?",(int(id),))
    db.commit()
    if not cur.rowcount:raise ValueError("Exemplo inexistente")
