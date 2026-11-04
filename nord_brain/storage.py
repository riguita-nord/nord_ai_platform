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
        CREATE TABLE IF NOT EXISTS intent_aliases (
            phrase TEXT PRIMARY KEY, intent TEXT NOT NULL, created INTEGER NOT NULL
        );
        """)
        # Archive one known V0.4 misinterpretation instead of deleting history.
        # Archived entries are never used to answer questions or for training replay.
        with self.db:
            self.db.execute("""
                UPDATE lessons SET origin='legacy_misinterpreted_instruction'
                WHERE lower(trim(question)) IN ('o que sabes', 'o que sabes?')
                AND origin='teacher'
                AND (lower(answer) LIKE 'quando te digo%'
                     OR lower(answer) LIKE 'quando eu disser%'
                     OR lower(answer) LIKE '%quero que me digas%')
            """)
        self.db.commit()

    def set_intent_alias(self, phrase: str, intent: str):
        from .conversation import normalize_request
        key=normalize_request(phrase)
        if not 2 <= len(key) <= 100 or intent!='self_knowledge':
            raise ValueError('Interpretação não suportada ou expressão inválida')
        with self.db:
            self.db.execute("""
                INSERT INTO intent_aliases(phrase,intent,created) VALUES(?,?,?)
                ON CONFLICT(phrase) DO UPDATE SET
                   intent=excluded.intent,created=excluded.created
            """, (key,intent,int(time.time())))

    def resolve_intent_alias(self, phrase: str):
        from .conversation import normalize_request
        key=normalize_request(phrase)
        if not 2 <= len(key) <= 100:
            return None
        row=self.db.execute(
            'SELECT intent FROM intent_aliases WHERE phrase=?',(key,)).fetchone()
        return row[0] if row else None

    def teacher_lesson_count(self):
        return self.db.execute(
            "SELECT COUNT(*) FROM lessons WHERE origin='teacher'").fetchone()[0]

    def recent_teacher_learnings(self, limit=4):
        """Display only teacher-supplied concepts and examples, never web guesses."""
        limit=max(1,min(int(limit),10))
        recent=[]
        seen=set()
        concepts=self.db.execute("""
            SELECT term,description FROM concepts WHERE origin='teacher'
            ORDER BY created DESC,term LIMIT ?
        """,(limit*2,)).fetchall()
        examples=self.db.execute("""
            SELECT question,answer FROM lessons WHERE origin='teacher'
            ORDER BY id DESC LIMIT ?
        """,(limit*3,)).fetchall()
        for term,desc in concepts:
            item=f'{term}: {desc}'
            norm=term.strip().casefold()
            if norm not in seen:
                recent.append(item.replace('\n',' ')[:160])
                seen.add(norm)
            if len(recent)>=limit:return recent
        for question,answer in examples:
            key=question.strip().casefold()
            if key not in seen:
                # A learned answer is quoted as an example, not as a new skill.
                recent.append(f'{question} → {answer}'.replace('\n',' ')[:160])
                seen.add(key)
            if len(recent)>=limit:break
        return recent

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
        row=self.db.execute("SELECT answer FROM lessons WHERE lower(question)=lower(?) AND origin!='legacy_misinterpreted_instruction' ORDER BY id DESC LIMIT 1",(question.strip(),)).fetchone()
        return row[0] if row else None

    def is_holdout(self, question: str) -> bool:
        return self.db.execute('SELECT 1 FROM holdouts WHERE lower(question)=lower(?)', (question.strip(),)).fetchone() is not None

    def lesson_count(self):
        return self.db.execute("SELECT COUNT(*) FROM lessons WHERE origin!='legacy_misinterpreted_instruction'").fetchone()[0]

    def lessons(self, limit=100):
        return self.db.execute("SELECT question,answer FROM lessons WHERE origin!='legacy_misinterpreted_instruction' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

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
        return self.db.execute("SELECT COUNT(DISTINCT lower(question)) FROM lessons WHERE origin!='legacy_misinterpreted_instruction'").fetchone()[0]

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
