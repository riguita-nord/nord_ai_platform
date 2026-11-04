"""Bounded local conversational context; never used as neural training labels.

Stores recent user/assistant exchanges so follow-up requests can refer to the
actual previous response, including offline stories. Data stays on the device.
"""
from __future__ import annotations

import re
import sqlite3
import time
import unicodedata
from pathlib import Path


def normalize(value: str) -> str:
    text = unicodedata.normalize('NFKD', value.casefold().strip())
    return ' '.join(''.join(char for char in text if not unicodedata.combining(char))
                    .strip(' .!?').split())


def followup_kind(message: str) -> str | None:
    """Recognize anaphoric requests only, not new topics or factual teaching."""
    t = normalize(message)
    t = re.sub(r'\s+por favor$', '', t)
    if not t:
        return None
    if re.fullmatch(r'(?:faz|escreve|cria|conta(?: me)?|quero|podes fazer)\s+'
                    r'(?:(?:uma|a|o|isso|a historia|o texto|uma versao)\s+)?'
                    r'(?:(?:versao|historia|texto)\s+)?(?:muito\s+)?'
                    r'(?:mais longa|mais comprida|mais extensa|maior)', t):
        return 'longer'
    if t in ('along[a] isso', 'alonga isso', 'desenvolve mais', 'desenvolve essa historia',
             'acrescenta mais detalhes', 'faz uma versao mais detalhada',
             'poe mais detalhes', 'faz mais detalhada', 'mais longa', 'mais comprida'):
        return 'longer'
    if t in ('continua', 'continua a historia', 'continua o conto', 'continua isso',
             'continua o texto', 'continua a narrativa', 'e depois', 'e depois o que aconteceu',
             'conta mais', 'quero saber o que acontece a seguir'):
        return 'continue'
    if t in ('muda o final', 'faz outro final', 'faz um final diferente',
             'altera o final', 'reescreve o final'):
        return 'ending'
    if t in ('reescreve', 'reescreve isso', 'reescreve a historia',
             'reescreve o texto', 'faz de outra maneira', 'faz outra versao'):
        return 'rewrite'
    if t in ('quem era ela', 'quem era ele', 'como se chamava a personagem',
             'qual era o nome da personagem', 'qual e o nome da personagem',
             'onde vivia', 'onde e que ela vivia', 'onde estava ela',
             'onde estava ele', 'o que encontrou', 'o que e que encontrou',
             'quem e a personagem', 'quem e o protagonista', 'e a personagem'):
        return 'reference'
    # Standalone references and naturally phrased follow-ups. The entire
    # utterance must match; new facts and explicit new questions stay intact.
    if re.fullmatch(
        r"(?:e\s+)?(?:o que achas|o que pensas|qual e a tua opiniao|"
        r"o que sugeres|alguma sugestao|tens ideias|quais as vantagens|"
        r"quais as desvantagens|qual e o problema|como melhoramos|"
        r"como se pode melhorar|como o podemos resolver|"
        r"que alternativas temos)(?:\s+(?:disso|sobre isso|para isso|dessa ideia))?", t):
        return 'discuss'
    if re.fullmatch(r'(?:e\s+)?(?:ele|ela|isso|esse|essa|deste|dessa|dele|dela)\??', t):
        return 'reference'
    if re.fullmatch(
        r"(?:e\s+)?(?:quem e|quem era|o que e|o que faz|"
        r"o que aconteceu a|como funciona|quando nasceu|onde fica|"
        r"qual e|porque e)(?:\s+(?:ele|ela|isso|esse|essa|desse|dela|dele))", t):
        return 'reference'
    if t in ('resume', 'resume isso','faz um resumo','resume o que disseste',
             'resume a resposta','faz mais curto','torna mais curto',
             'simplifica','simplifica isso','explica de forma simples'):
        return 'shorter'
    if t in ('reescreve em ingles','traduz para ingles','traduz isso para ingles',
             'traduz para espanhol','traduz para frances',
             'faz mais formal','faz mais informal','muda o tom'):
        return 'rewrite'
    if t in ('da me ideias', 'tens alguma ideia', 'sugere alternativas',
             'e outras possibilidades', 'que outras possibilidades existem'):
        return 'discuss'
    if t in ('e por que', 'e porque', 'mas porque', 'mas por que',
             'podes explicar porque', 'porque e que isso acontece',
             'explica a razao'):
        return 'explain'
    if t in ('explica melhor', 'explica isso melhor', 'podes explicar melhor',
             'podes explicar isso melhor', 'desenvolve essa ideia',
             'explica com mais detalhes', 'fala mais sobre isso',
             'da me um exemplo disso', 'da outro exemplo',
             'da me mais exemplos', 'porque', 'por que', 'como assim'):
        return 'explain'
    return None


class DialogueMemory:
    """A separate, size-limited SQLite history that cannot contaminate lessons."""
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / 'dialogue.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=5)
        self.db.execute('''CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY, question TEXT NOT NULL, reply TEXT NOT NULL,
            origin TEXT NOT NULL, focus TEXT NOT NULL, created INTEGER NOT NULL,
            topic TEXT NOT NULL DEFAULT '')''')
        # Migrate the previous dialogue store without touching brain lessons.
        cols={row[1] for row in self.db.execute('PRAGMA table_info(turns)')}
        if 'topic' not in cols:
            self.db.execute("ALTER TABLE turns ADD COLUMN topic TEXT NOT NULL DEFAULT ''")
        self.db.commit()

    def add(self, question: str, reply: str, origin: str, focus: str, topic: str = ''):
        if focus not in ('story', 'book', 'dialogue', 'math', 'research', 'system'):
            focus = 'dialogue'
        with self.db:
            self.db.execute('INSERT INTO turns(question,reply,origin,focus,created,topic) VALUES (?,?,?,?,?,?)',
                            (question[:1800],reply[:9000],origin[:150],focus,int(time.time()),topic[:250]))
            self.db.execute('DELETE FROM turns WHERE id NOT IN '
                            '(SELECT id FROM turns ORDER BY id DESC LIMIT 24)')

    def latest(self) -> dict | None:
        row = self.db.execute('SELECT question,reply,origin,focus,topic FROM turns '
                              'ORDER BY id DESC LIMIT 1').fetchone()
        return dict(zip(('question','reply','origin','focus','topic'),row)) if row else None

    def recent(self, count: int = 3) -> list[dict]:
        rows = self.db.execute('SELECT question,reply,origin,focus,topic FROM turns '
                               'ORDER BY id DESC LIMIT ?', (min(8,max(1,count)),)).fetchall()
        return [dict(zip(('question','reply','origin','focus','topic'),row)) for row in reversed(rows)]

    def context_messages(self) -> list[dict[str, str]]:
        """Bounded ordered role history for the local language model.

        All response origins (math, web, books, ordinary chat) are included,
        rather than only messages previously generated by the LLM. Text from
        a webpage remains conversation data, never an instruction or label.
        """
        rows=self.recent(5)
        messages=[]
        for row in rows:
            messages.append({'role':'user','content':row['question'][:380]})
            reply=row['reply']
            if len(reply)>850:
                reply=reply[:500]+'\n[texto anterior abreviado]\n'+reply[-300:]
            messages.append({'role':'assistant','content':reply})
        return messages

    def clear(self):
        with self.db:
            self.db.execute('DELETE FROM turns')

    def close(self):
        self.db.close()
