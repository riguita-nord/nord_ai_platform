"""Persistent teaching events, concepts, sourced observations and frozen tests."""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path


class Memory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL,
            origin TEXT NOT NULL, created INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS lessons_question ON lessons(question);
        CREATE TABLE IF NOT EXISTS concepts (
            term TEXT PRIMARY KEY, description TEXT NOT NULL, origin TEXT NOT NULL,
            created INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY, topic TEXT NOT NULL, title TEXT NOT NULL,
            url TEXT NOT NULL, summary TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0,
            created INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS holdouts (
            id INTEGER PRIMARY KEY, question TEXT UNIQUE NOT NULL, answer TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY, event TEXT NOT NULL, value REAL NOT NULL, created INTEGER NOT NULL);
        ''')
        self.db.commit()

    def teach(self, question: str, answer: str, origin: str = 'teacher') -> int:
        question,answer=question.strip(),answer.strip()
        if not 1 <= len(question) <= 250 or not 1 <= len(answer) <= 350:
            raise ValueError('Exemplos limitados a 250/350 caracteres')
        with self.db:
            cur=self.db.execute('INSERT INTO lessons(question,answer,origin,created) VALUES(?,?,?,?)',
                            (question,answer,origin,int(time.time())))
        return int(cur.lastrowid)

    def exact(self, question: str):
        row=self.db.execute('SELECT answer FROM lessons WHERE lower(question)=lower(?) ORDER BY id DESC LIMIT 1',(question.strip(),)).fetchone()
        return row[0] if row else None

    def is_holdout(self, question: str) -> bool:
        return self.db.execute('SELECT 1 FROM holdouts WHERE lower(question)=lower(?)', (question.strip(),)).fetchone() is not None

    def lesson_count(self):
        return self.db.execute('SELECT COUNT(*) FROM lessons').fetchone()[0]

    def lessons(self, limit=100):
        return self.db.execute('SELECT question,answer FROM lessons ORDER BY id DESC LIMIT ?', (limit,)).fetchall()

    def concept(self, term: str, definition: str, origin='teacher'):
        term,definition=term.strip().lower(),definition.strip()
        if not 1 <= len(term) <= 60 or not 1 <= len(definition) <= 350:
            raise ValueError('Conceito demasiado longo ou vazio')
        with self.db:
            self.db.execute('INSERT INTO concepts VALUES(?,?,?,?) ON CONFLICT(term) DO UPDATE SET description=excluded.description,origin=excluded.origin,created=excluded.created',
                            (term,definition,origin,int(time.time())))

    def lookup_concept(self,term):
        row=self.db.execute('SELECT description FROM concepts WHERE term=?',(term.strip().lower(),)).fetchone()
        return row[0] if row else None

    def observation(self,topic,title,url,summary):
        with self.db:
            self.db.execute('INSERT INTO observations(topic,title,url,summary,created) VALUES(?,?,?,?,?)',
                (topic,title,url,summary,int(time.time())))

    def holdout(self,question,answer):
        if not 1 <= len(question) <= 250 or not 1 <= len(answer) <= 350:
            raise ValueError('Teste inválido')
        if self.db.execute('SELECT 1 FROM lessons WHERE lower(question)=lower(?)', (question.strip(),)).fetchone():
            raise ValueError('Esta pergunta já foi ensinada; usa um teste novo')
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO holdouts(question,answer) VALUES(?,?)',(question,answer))

    def tests(self,limit=10):
        return self.db.execute('SELECT question,answer FROM holdouts ORDER BY id LIMIT ?', (limit,)).fetchall()

    def metric(self,event,value):
        with self.db:
            self.db.execute('INSERT INTO metrics(event,value,created) VALUES(?,?,?)',(event,float(value),int(time.time())))

    def close(self):
        self.db.close()
