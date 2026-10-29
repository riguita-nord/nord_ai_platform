"""Lightweight deterministic answer quality checks for small models."""
import re
import unicodedata

def words(text):
    text=unicodedata.normalize("NFKD",str(text).lower())
    text="".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]{3,}",text)

def is_explanation(question):
    return bool(re.search(r"\b(o que e|quem e|como funciona|explica|define|para que serve)\b"," ".join(words(question))))

def needs_revision(question,answer):
    if not is_explanation(question):return False
    a=words(answer)
    if len(a)<35:return True
    q=set(words(question))
    if not q:return False
    # Fails tautologies like "QBCore is a framework where QBCore is the main component".
    if len(a)<65 and len(set(a))<max(9,len(a)//3):return True
    if "framework" in q or "qbcore" in q:
        specific={"jogadores","servidores","fivem","scripts","empregos","dinheiro",
                  "eventos","inventario","dados","recursos","funcoes","modulos"}
        if not specific.intersection(set(a)) and len(a)<75:return True
    return False
