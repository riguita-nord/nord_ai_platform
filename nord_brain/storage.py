"""Persistent teaching events, concepts, sourced observations and frozen tests."""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path


class Memory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
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
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS unknown_topics (
            topic TEXT PRIMARY KEY, attempts INTEGER NOT NULL DEFAULT 0,
            searched INTEGER NOT NULL DEFAULT 0, created INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS autonomy_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS autonomy_events (
            id INTEGER PRIMARY KEY, kind TEXT NOT NULL, detail TEXT NOT NULL,
            value REAL, created INTEGER NOT NULL);
        ''')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS reasoning_facts (
            subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
            statement TEXT NOT NULL, origin TEXT NOT NULL, created INTEGER NOT NULL,
            PRIMARY KEY (subject, predicate, object)
        );
        ''')
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS intent_aliases (phrase TEXT PRIMARY KEY,
            intent TEXT NOT NULL, created INTEGER NOT NULL);
        """)
        # Quarantine only narrowly recognisable old parser mistakes. The
        # original rows remain in SQLite for inspection/recovery, but are not
        # replayed as correct teacher labels.
        with self.db:
            self.db.execute("""
                UPDATE lessons SET origin='legacy_misparsed_math'
                WHERE origin='teacher' AND (
                  lower(trim(question)) LIKE 'e diz-me quanto%'
                  OR lower(trim(answer)) LIKE 'isto matematica 2+2%'
                )
            """)
            self.db.execute("""
                UPDATE lessons SET origin='legacy_uncertain_reply'
                WHERE origin='teacher' AND lower(trim(answer)) IN
                  ('agora nao consigo','agora não consigo')
            """)
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS math_hypotheses (
            question TEXT PRIMARY KEY, expected TEXT NOT NULL,
            seed TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
            prediction_before TEXT, created INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS math_hypotheses_state ON math_hypotheses(state);
        CREATE TABLE IF NOT EXISTS source_learning (
            url TEXT PRIMARY KEY, topic TEXT NOT NULL,
            lesson_id INTEGER NOT NULL, created INTEGER NOT NULL);
        ''')
        self.db.commit()

    def queue_math_hypotheses(self, seed: str, examples):
        inserted=0
        for question, expected in list(examples)[:48]:
            if self.is_holdout(question) or self.exact(question) is not None:
                continue
            with self.db:
                row=self.db.execute('''INSERT OR IGNORE INTO math_hypotheses
                    (question,expected,seed,state,created) VALUES (?,?,?,'pending',?)''',
                    (question,expected,seed[:100],int(time.time())))
                inserted+=int(row.rowcount==1)
        return inserted

    def pending_math_hypotheses(self, limit=3):
        return self.db.execute('''SELECT question,expected FROM math_hypotheses
          WHERE state='pending' ORDER BY created,question LIMIT ?''',
          (max(1,min(int(limit),12)),)).fetchall()

    def mark_math_hypothesis(self, question, prediction, state):
        if state not in ('trained','rejected'):
            raise ValueError('Estado de hipótese inválido')
        with self.db:
            self.db.execute('''UPDATE math_hypotheses SET state=?,prediction_before=?
                               WHERE question=? AND state='pending' ''',
                            (state,str(prediction)[:40],question))

    def hypothesis_stats(self):
        return dict(self.db.execute('''SELECT state,COUNT(*) FROM math_hypotheses
                                    GROUP BY state''').fetchall())

    def source_was_learned(self,url):
        return self.db.execute('SELECT 1 FROM source_learning WHERE url=?',(url,)).fetchone() is not None

    def mark_source_learned(self, url, topic, lesson_id):
        with self.db:
            self.db.execute('''INSERT OR IGNORE INTO source_learning
               (url,topic,lesson_id,created) VALUES (?,?,?,?)''',
               (url,topic,int(lesson_id),int(time.time())))

    def source_count(self):
        return self.db.execute('SELECT COUNT(*) FROM source_learning').fetchone()[0]

    def set_intent_alias(self,phrase,intent):
        from .conversation import normalize_request
        key=normalize_request(phrase)
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO intent_aliases VALUES(?,?,0)',(key,intent))

    def resolve_intent_alias(self,phrase):
        from .conversation import normalize_request
        row=self.db.execute('SELECT intent FROM intent_aliases WHERE phrase=?',
                            (normalize_request(phrase),)).fetchone()
        return row[0] if row else None

    def teacher_lesson_count(self):
        return self.db.execute("SELECT COUNT(*) FROM lessons WHERE origin='teacher'").fetchone()[0]

    def recent_teacher_learnings(self,limit=4):
        rows=self.db.execute("SELECT term,description FROM concepts WHERE origin='teacher' LIMIT ?",(limit,)).fetchall()
        return [f'{t}: {d}' for t,d in rows]

    def add_reasoning_fact(self, subject, predicate, obj, statement, origin='teacher'):
        if predicate not in ('is_a','has') or origin!='teacher':
            raise ValueError('Apenas premissas confirmadas pelo professor')
        with self.db:
            self.db.execute('''INSERT INTO reasoning_facts
                (subject,predicate,object,statement,origin,created) VALUES (?,?,?,?,?,?)
                ON CONFLICT(subject,predicate,object) DO UPDATE SET
                statement=excluded.statement,created=excluded.created''',
                (subject,predicate,obj,statement,origin,int(time.time())))

    def reasoning_facts(self, limit=300):
        return self.db.execute('''SELECT subject,predicate,object,statement
            FROM reasoning_facts ORDER BY created,subject,predicate,object LIMIT ?''',
            (limit,)).fetchall()

    def reasoning_fact_count(self):
        return self.db.execute('SELECT COUNT(*) FROM reasoning_facts').fetchone()[0]

    def teach(self, question: str, answer: str, origin: str = 'teacher') -> int:
        question,answer=question.strip(),answer.strip()
        if not 1 <= len(question) <= 250 or not 1 <= len(answer) <= 500:
            raise ValueError('Exemplos limitados a 250/500 caracteres')
        with self.db:
            cur=self.db.execute('INSERT INTO lessons(question,answer,origin,created) VALUES(?,?,?,?)',
                            (question,answer,origin,int(time.time())))
        return int(cur.lastrowid)

    def exact(self, question: str):
        row=self.db.execute("SELECT answer FROM lessons WHERE lower(question)=lower(?) AND origin NOT LIKE 'legacy_%' ORDER BY id DESC LIMIT 1",(question.strip(),)).fetchone()
        return row[0] if row else None

    def is_holdout(self, question: str) -> bool:
        return self.db.execute('SELECT 1 FROM holdouts WHERE lower(question)=lower(?)', (question.strip(),)).fetchone() is not None

    def lesson_count(self):
        return self.db.execute("SELECT COUNT(*) FROM lessons WHERE origin NOT LIKE 'legacy_%'").fetchone()[0]

    def lessons(self, limit=100):
        return self.db.execute("SELECT question,answer FROM lessons WHERE origin NOT LIKE 'legacy_%' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

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
        if not 1 <= len(question) <= 250 or not 1 <= len(answer) <= 500:
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

    def unique_lesson_count(self):
        return self.db.execute("SELECT COUNT(DISTINCT lower(question)) FROM lessons WHERE origin NOT LIKE 'legacy_%'").fetchone()[0]

    def get_meta(self,key,default=None):
        row=self.db.execute('SELECT value FROM autonomy_meta WHERE key=?',(key,)).fetchone()
        return row[0] if row else default

    def set_meta(self,key,value):
        with self.db:
            self.db.execute('INSERT INTO autonomy_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,str(value)))

    def unknown(self, topic):
        topic=topic.strip().strip('?.!')[:90]
        if len(topic)<2:return
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO unknown_topics(topic,created) VALUES(?,?)',
                            (topic,int(time.time())))

    def next_unknown(self):
        row=self.db.execute('SELECT topic FROM unknown_topics WHERE searched=0 ORDER BY created LIMIT 1').fetchone()
        return row[0] if row else None

    def unknown_searched(self, topic):
        with self.db:
            self.db.execute('UPDATE unknown_topics SET searched=1,attempts=attempts+1 WHERE topic=?',(topic,))

    def event(self,kind,detail,value=None):
        with self.db:
            self.db.execute('INSERT INTO autonomy_events(kind,detail,value,created) VALUES(?,?,?,?)',
                        (kind,detail,float(value) if value is not None else None,int(time.time())))

    def recent_events(self,limit=10):
        return self.db.execute('SELECT kind,detail,value FROM autonomy_events ORDER BY id DESC LIMIT ?',(limit,)).fetchall()

    def recent_losses(self,count=3):
        return [r[0] for r in self.db.execute(
            "SELECT value FROM autonomy_events WHERE kind='eval_loss' ORDER BY id DESC LIMIT ?",(count,)).fetchall()][::-1]

    def observation_for_topic(self, topic):
        return self.db.execute('SELECT title,url,summary FROM observations WHERE lower(topic)=lower(?) ORDER BY id DESC LIMIT 1',(topic,)).fetchone()

    def close(self):
        self.db.close()
