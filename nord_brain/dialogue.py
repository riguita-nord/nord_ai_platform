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
    # Elliptical edits: users often add a constraint to the previous answer
    # ("com capítulos e etc.", "sem spoilers", "em pontos", "com diálogos").
    # These are contextual requests, NEVER teacher labels. The active topic
    # is checked by Brain before routing; this parser does not invent one.
    if len(t) <= 150:
        if (re.search(r'\bcapitulos?\b', t)
                and re.match(r'^(?:(?:e|agora|entao)\s+)?'
                             r'(?:(?:com|em|por|dividid[oa]s?\s+em|'
                             r'organiza(?:\s+isso)?\s+em|'
                             r'transforma(?:\s+isso)?\s+em|'
                             r'faz(?:\s+isso)?\s+(?:com|em)|'
                             r'quero(?:\s+isso)?\s+(?:com|em))\b)',t)):
            return 'chapters'
        if re.match(r'^(?:(?:e|agora|entao)\s+)?(?:com|sem|em|por|'
                    r'no\s+estilo\s+de)\s+\S',t) and not re.match(
                    r'^(?:com\s+(?:quem|o\s+que)|em\s+(?:que|qual|quanto|onde)|'
                    r'por\s+(?:que|quem)|sem\s+(?:quem|o\s+que))\b',t):
            return 'refine'
        if re.match(r'^(?:faz|escreve|torna|organiza|reescreve|transforma|'
                    r'apresenta|mostra|mete|poe|coloca)(?:\s+(?:isso|a\s+historia|'
                    r'o\s+texto|a\s+resposta))?\s+(?:com|sem|em|por)\s+\S',t):
            return 'refine'
    if re.fullmatch(r'(?:faz|escreve|cria|conta(?: me)?|quero|podes fazer)\s+'
                    r'(?:(?:uma|a|o|isso|a historia|o texto|uma versao)\s+)?'
                    r'(?:(?:versao|historia|texto)\s+)?(?:muito\s+)?'
                    r'(?:mais longa|mais comprida|mais extensa|maior)', t):
        return 'longer'
    # Short, natural changes to what was just discussed are follow-ups, not
    # answers to an earlier teacher question. Match the intention, not only
    # exact memorised sentences (e.g. "faz mais extenso").
    if len(t) <= 145 and re.match(
        r'^(?:(?:e|entao)\s+)?(?:faz|fazer|escreve|escrever|conta|contar|'
        r'podes|consegues|poderias|consegue|quero|aumenta|alonga|desenvolve|'
        r'torna|transforma|reformula|expande|uma|um|mais)\b', t):
        if re.search(r'\b(?:maior|mais\s+(?:long[ao]|comprid[ao]|extens[ao]|'
                     r'detalhad[ao]|desenvolvid[ao]|texto|conteudo|pormenores)|'
                     r'com\s+mais\s+(?:detalhes|pormenores))\b', t):
            return 'longer'
        if re.search(r'\b(?:mais\s+(?:curt[ao]|resumid[ao]|simples)|menor)\b', t):
            return 'shorter'
    if re.fullmatch(r'(?:(?:podes|consegues|poderias)\s+)?(?:continuar|continua)'
                    r'(?:\s+(?:a\s+historia|o\s+texto|a\s+explicacao|dai|isso))?', t):
        return 'continue'
    if re.fullmatch(r'(?:(?:podes|consegues|poderias)\s+)?(?:explica|explicar)'
                    r'(?:\s+(?:isso|aquilo|me|nos))?\s+(?:melhor|outra\s+vez)', t):
        return 'explain'
    if re.search(r'\b(?:isso|disso|aquilo|daquilo|dessa\s+ideia|'
                 r'desse\s+projeto|que\s+falamos|anterior)\b', t) and (
            t.startswith(('e ', 'mas ', 'entao ', 'como ', 'porque ', 'por que ',
                          'qual ', 'quais ', 'o que ', 'podes ', 'consegues ', 'faz ',
                          'mostra ', 'explica ', 'diz ', 'retoma ', 'volta '))):
        return 'discuss'
    # Short questions prefaced by "e" normally build on the current topic,
    # unless they introduce a separate numerical exercise.
    if len(t) <= 150 and re.match(r'^e\s+(?:como|porque|por que|qual|quais|'
                                  r'o que|se|entao|agora|ele|ela|isso)\b',t) \
            and not re.search(r'\d\s*[+*/-]\s*\d',t):
        return 'discuss'
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
            # Keep a searchable history beyond the model's context window.
            # Old turns remain local and are never used as teacher labels.
            self.db.execute('DELETE FROM turns WHERE id NOT IN '
                            '(SELECT id FROM turns ORDER BY id DESC LIMIT 480)')

    def latest(self) -> dict | None:
        row = self.db.execute('SELECT question,reply,origin,focus,topic FROM turns '
                              'ORDER BY id DESC LIMIT 1').fetchone()
        return dict(zip(('question','reply','origin','focus','topic'),row)) if row else None

    def recent(self, count: int = 3) -> list[dict]:
        rows = self.db.execute('SELECT question,reply,origin,focus,topic FROM turns '
                               'ORDER BY id DESC LIMIT ?', (min(480,max(1,count)),)).fetchall()
        return [dict(zip(('question','reply','origin','focus','topic'),row)) for row in reversed(rows)]

    @staticmethod
    def _terms(text: str) -> set[str]:
        """Cheap CPU-only lexical retrieval; never mistakes it for semantics."""
        stop={'a','o','os','as','de','da','do','dos','das','um','uma','uns','umas',
              'e','ou','que','se','eu','tu','ele','ela','isso','esse','essa',
              'mais','menos','para','por','sobre','com','sem','no','na','nos','nas',
              'me','te','diz','podes','consegues','faz','fazer','qual','como',
              'quero','agora','depois','entao','esta','estamos','foi','era',
              'aquela','aquele','anterior','falamos','falar','texto','resposta'}
        return {w for w in re.findall(r'[a-z0-9]{3,}', normalize(text)) if w not in stop}

    def latest_story(self) -> dict | None:
        """Recover the actual narrative after intervening questions about it."""
        for row in reversed(self.recent(24)):
            if row['focus']=='story' and len(row['reply']) >= 240 and (
                    'narrativa' in row['origin'] or 'ficção contextual' in row['origin']
                    or 'linguagem pré-treinada' in row['origin']):
                return row
        return None

    def context_messages(self, query: str = '') -> list[dict[str, str]]:
        """Relevant old exchanges + recent conversation, within a CPU-sized budget.

        The selection always considers the stored dialogue, not just the last
        reply. Long-term lessons and retrieved pages are not promoted to system
        instructions, and no conversational history is used as a training label.
        """
        # A short rolling window preserves recent dialogue even after topic
        # changes. A few older turns are retrieved by relevance to the current
        # request (and the active topic for references like 'isso').
        latest = self.db.execute(
            'SELECT id,question,reply,origin,focus,topic FROM turns '
            'ORDER BY id DESC LIMIT 8').fetchall()
        earlier = self.db.execute(
            'SELECT id,question,reply,origin,focus,topic FROM turns '
            'WHERE id < ? ORDER BY id DESC LIMIT 472',
            (latest[-1][0],)).fetchall() if latest else []
        terms=self._terms(query)
        if not terms and latest:
            terms=self._terms(latest[0][5] or latest[0][1])
        # Anaphoric phrases borrow the current subject even when they contain
        # no topic terms themselves; new independent topics do not.
        if latest and (not terms or re.search(
                r'\b(?:isso|disso|aquilo|daquilo|ele|ela|anterior)\b',normalize(query))):
            terms |= self._terms(latest[0][5])
        scored=[]
        for row in earlier:
            if not terms:break
            question_terms=self._terms(row[1])
            topic_terms=self._terms(row[5])
            answer_terms=self._terms(row[2][:650])
            score=4*len(terms & question_terms)+3*len(terms & topic_terms)+len(terms & answer_terms)
            if score:scored.append((score,row[0],row))
        # Include up to three older, related exchanges, maintaining chronology.
        matching=[item[2] for item in sorted(scored,reverse=True)[:3]]
        rows=sorted(matching+latest,key=lambda item:item[0])
        messages=[]
        for _id,question,reply,_origin,_focus,_topic in rows:
            messages.append({'role':'user','content':question[:340]})
            if len(reply)>800:
                reply=reply[:440]+'\n[resposta anterior abreviada]\n'+reply[-270:]
            messages.append({'role':'assistant','content':reply})
        return messages

    def clear(self):
        with self.db:
            self.db.execute('DELETE FROM turns')

    def close(self):
        self.db.close()
