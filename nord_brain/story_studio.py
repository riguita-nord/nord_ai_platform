"""Persistent, chapter-by-chapter fiction studio for CPU-constrained Nord Brain.

Fiction is never a factual teaching example. A compact story bible and saved
chapter recaps let a small local LLM continue a book without loading the whole
manuscript into its context window. The offline path is a transparent procedural
fallback, not proof of neural creativity.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import tempfile
import time
import unicodedata

from .creative import CHARACTERS, PLACES, DISCOVERIES, CONFLICTS


@dataclass(frozen=True)
class BookCommand:
    action: str
    topic: str = ''
    amount: int = 1


def _ascii(value: str) -> str:
    norm=unicodedata.normalize('NFKD',value.casefold())
    return ''.join(c for c in norm if not unicodedata.combining(c))


def parse_book_command(message: str) -> BookCommand | None:
    """Natural Portuguese commands for long fiction, not generic research."""
    value=' '.join(message.strip().rstrip(' .!?').split())
    normal=_ascii(value)
    if not normal:return None
    if normal in ('lista livros','lista historias','mostra livros','mostra historias','os meus livros'):
        return BookCommand('list')
    if normal in ('estado do livro','estado da historia','como vai o livro','plano do livro','plano da historia'):
        return BookCommand('status')
    if normal in ('exporta o livro','exportar livro','guarda o livro','descarrega o livro'):
        return BookCommand('export')
    match=re.fullmatch(r'(?:mostra|le|ler|rele|reler)\s+(?:o\s+)?capitulo\s+(\d{1,2})',normal)
    if match:return BookCommand('read',amount=int(match[1]))
    match=re.fullmatch(r'(?:abre|seleciona|retoma)\s+(?:o\s+)?livro\s+(\d{1,3})',normal)
    if match:return BookCommand('select',amount=int(match[1]))
    match=re.fullmatch(r'(?:continua|continuar|escreve|gera|cria)(?:\s+(?:a|o))?\s+(?:historia|livro|narrativa)(?:\s+(?:por|com|mais))?\s+(\d{1,2})\s+capitulos?',normal)
    if match:return BookCommand('continue',amount=int(match[1]))
    extra=re.fullmatch(r'(?:continua|escreve|gera|faz)(?:\s+(?:por|com))?\s+mais\s+(\d{1,2})\s+capitulos?',normal)
    if extra:return BookCommand('continue',amount=int(extra[1]))
    if normal in ('continua','continuar','continua historia','continua a historia','continua o livro',
                  'continua esta historia','continua este livro','proximo capitulo',
                  'escreve o proximo capitulo','gera o proximo capitulo','mais um capitulo',
                  'continua a narrativa','escreve mais'):
        return BookCommand('continue')
    # The topic is captured from the original text to retain accents/casing.
    match=re.fullmatch(
        r'(?:conta(?:-me| me)?|inventa|cria|escreve|faz)\s+(?:(?:uma|um|o|a)\s+)?'
        r'(?:hist[oó]ria|hitoria|est[oó]ria|narrativa)\s+(?:longa|grande|maior|mais longa|comprida)'
        r'(?:\s+(?:com|de)\s+(\d{1,2})\s+cap[ií]tulos?)?'
        r'(?:\s+(?:sobre|acerca de)\s+(.+))?',value,re.I)
    if match:return BookCommand('start',topic=match[2] or 'uma aventura inesperada',
                                amount=int(match[1]) if match[1] else 10)
    match=re.fullmatch(
        r'(?:cria|escreve|inventa|faz|come[cç]a)\s+(?:(?:um|o)\s+)?(?:livro|romance)'
        r'(?:\s+(?:com|de)\s+(\d{1,2})\s+cap[ií]tulos?)?'
        r'(?:\s+(?:sobre|acerca de)\s+(.+))?',value,re.I)
    if match:return BookCommand('start',topic=match[2] or 'uma aventura inesperada',
                                amount=int(match[1]) if match[1] else 10)
    return None


STAGES=(
    ('A primeira pista', 'Apresentar o protagonista e a rotina antes da descoberta inesperada.'),
    ('A porta entreaberta', 'Algo muda; o protagonista escolhe investigar em vez de ignorar o sinal.'),
    ('Um aliado improvável', 'Introduzir uma ajuda com interesses próprios e uma pista ambígua.'),
    ('Uma hipótese perigosa', 'A personagem formula uma teoria e experimenta uma solução limitada.'),
    ('O preço da escolha', 'A experiência tem consequências e afeta uma relação importante.'),
    ('A falsa resposta', 'Uma descoberta aparentemente definitiva revela-se incompleta.'),
    ('O que ficou escondido', 'A verdadeira motivação de uma personagem altera o plano.'),
    ('Quando tudo falha', 'O protagonista perde um recurso e tem de escolher o que preservar.'),
    ('A decisão difícil', 'Várias saídas são comparadas e nenhuma é isenta de custos.'),
    ('O confronto', 'Confrontar o conflito central usando o que foi aprendido.'),
    ('A mudança', 'Mostrar as consequências concretas da decisão tomada.'),
    ('Um novo horizonte', 'Encerrar o arco narrativo e deixar uma consequência emocional.')
)


def _chapter_outline(total: int, number: int) -> tuple[str,str]:
    # Preserve introduction and resolution at both extremes, even in long books.
    position=round((number-1)*(len(STAGES)-1)/(total-1))
    title,beat=STAGES[position]
    if total>len(STAGES) and number>1 and round((number-2)*(len(STAGES)-1)/(total-1))==position:
        title+=' — Novos sinais'
        beat+=' Desenvolver este conflito por outro ponto de vista, sem repetir a cena anterior.'
    return title,beat


def _clean_text(value: str,limit: int) -> str:
    return re.sub(r'[\x00-\x1f<>]', ' ',value).strip()[:limit]


class StoryStudio:
    def __init__(self,root: Path):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'stories.sqlite3',check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY, title TEXT NOT NULL, theme TEXT NOT NULL,
                planned INTEGER NOT NULL, seed INTEGER NOT NULL,
                bible TEXT NOT NULL, created INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chapters (
                book_id INTEGER NOT NULL, number INTEGER NOT NULL,
                title TEXT NOT NULL, body TEXT NOT NULL, recap TEXT NOT NULL,
                created INTEGER NOT NULL,
                PRIMARY KEY (book_id,number),
                FOREIGN KEY (book_id) REFERENCES books(id)
            );
            CREATE TABLE IF NOT EXISTS book_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        self.db.commit()

    def create(self,theme: str,chapters: int=10) -> dict:
        theme=_clean_text(theme,160)
        if not 3<=chapters<=24:raise ValueError('O livro deve ter entre 3 e 24 capítulos.')
        if len(theme)<3:raise ValueError('Descreve o tema do livro com pelo menos três caracteres.')
        rng=random.SystemRandom()
        seed=rng.randrange(1,2**31)
        r=random.Random(seed)
        protagonist=r.choice(CHARACTERS)
        setting=r.choice(PLACES)
        discovery=r.choice(DISCOVERIES)
        conflict=r.choice(CONFLICTS)
        bible={'protagonist':protagonist,'setting':setting,'discovery':discovery,
               'conflict':conflict,'theme':theme}
        title=f'Os caminhos de {theme[:54]}'
        with self.db:
            cur=self.db.execute('INSERT INTO books(title,theme,planned,seed,bible,created) VALUES(?,?,?,?,?,?)',
                (title,theme,chapters,seed,json.dumps(bible,ensure_ascii=False),int(time.time())))
            book_id=int(cur.lastrowid)
            self.db.execute('INSERT OR REPLACE INTO book_meta(key,value) VALUES(?,?)',('active',str(book_id)))
        return self.active()

    def adopt_story(self, narrative: str, *, chapters: int = 10) -> dict:
        """Turn the current short story into a persistent book, not a NEW plot.

        Preserve every existing paragraph in order. Split the narrative into
        three opening chapters and save its actual protagonist/setting/discovery
        as the story bible so later chapters continue the same world.
        """
        if not 4 <= chapters <= 24:
            raise ValueError('O livro deve ter entre 4 e 24 capítulos.')
        text=narrative.strip()
        parts=[part.strip() for part in re.split(r'\n\s*\n',text) if part.strip()]
        if len(parts)<2 or len(text)<180:
            raise ValueError('Não há uma história completa para transformar em livro.')
        title=parts[0].splitlines()[0].strip()[:110]
        if len(parts[0])<130 and len(parts)>2 and title and not title.endswith(('.', '!', '?')):
            paragraphs=parts[1:]
        else:
            title='A história que começámos'
            paragraphs=parts
        if len(paragraphs)<2:
            raise ValueError('Preciso de mais conteúdo antes de criar capítulos.')

        from .story_revision import _elements
        protagonist, setting, discovery=_elements(text)
        if protagonist=='a protagonista':
            protagonist=paragraphs[0].split(',',1)[0].strip()[:70] or protagonist
        conflict='o mistério apresentado nos primeiros acontecimentos'
        for paragraph in paragraphs:
            match=re.search(r'\bpercebeu que (.+?)[.!?]',paragraph,re.I|re.S)
            if match:
                conflict=match.group(1).strip()[:240]
                break

        book=self.create(title,chapters=chapters)
        bible={'protagonist':protagonist,'setting':setting,
               'discovery':discovery,'conflict':conflict,
               'theme':title,'origin':'adopted_conversation_story'}
        chunks=[]
        count=min(3,len(paragraphs))
        for index in range(count):
            # Consecutive balanced chunks, so not a single paragraph is lost.
            start=(index*len(paragraphs))//count
            end=((index+1)*len(paragraphs))//count
            chunks.append('\n\n'.join(paragraphs[start:end]))
        chapter_titles=('O início', 'O enigma', 'A escolha')
        now=int(time.time())
        with self.db:
            self.db.execute('UPDATE books SET title=?,theme=?,bible=? WHERE id=?',
                            (title,title,json.dumps(bible,ensure_ascii=False),book['id']))
            for n,body in enumerate(chunks,1):
                recap=(f'Início: {body[:200].strip()} '
                       f'Final: {body[-250:].strip()}')[:550]
                self.db.execute(
                    'INSERT INTO chapters(book_id,number,title,body,recap,created) '
                    'VALUES(?,?,?,?,?,?)',
                    (book['id'],n,chapter_titles[n-1],body,recap,now))
        return {'book':self.active(),
                'chapters':[{'number':n,'title':chapter_titles[n-1],'body':body}
                            for n,body in enumerate(chunks,1)]}

    def active(self) -> dict | None:
        row=self.db.execute('''SELECT b.id,b.title,b.theme,b.planned,b.seed,b.bible,
                                  (SELECT COUNT(*) FROM chapters c WHERE c.book_id=b.id)
            FROM books b WHERE b.id=(SELECT CAST(value AS INTEGER) FROM book_meta WHERE key='active')''').fetchone()
        if row is None:return None
        keys=('id','title','theme','planned','seed','bible','written')
        data=dict(zip(keys,row))
        data['bible']=json.loads(data['bible'])
        return data

    def select(self,book_id: int) -> dict | None:
        if not self.db.execute('SELECT 1 FROM books WHERE id=?',(book_id,)).fetchone():
            return None
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO book_meta(key,value) VALUES(?,?)',('active',str(book_id)))
        return self.active()

    def list_books(self,limit=20):
        return self.db.execute('''SELECT b.id,b.title,b.planned,
                 (SELECT COUNT(*) FROM chapters c WHERE c.book_id=b.id)
                 FROM books b ORDER BY b.id DESC LIMIT ?''',(min(max(limit,1),50),)).fetchall()

    def chapter(self,number:int) -> tuple | None:
        book=self.active()
        if not book:return None
        return self.db.execute('SELECT title,body FROM chapters WHERE book_id=? AND number=?',
                               (book['id'],number)).fetchone()

    def _context(self,book:dict,number:int) -> str:
        rows=self.db.execute('SELECT number,title,recap FROM chapters WHERE book_id=? ORDER BY number DESC LIMIT 4',
                             (book['id'],)).fetchall()
        if not rows:return 'É o primeiro capítulo. Não existem acontecimentos anteriores.'
        history='\n'.join(f'Capítulo {n}: {title} — {recap[:190]}' for n,title,recap in reversed(rows))
        last=self.db.execute('SELECT body FROM chapters WHERE book_id=? AND number=?',
                             (book['id'],number-1)).fetchone()
        return history+'\nÚltimas linhas anteriores: '+(last[0][-440:] if last else '')

    def _prompt(self,book:dict,number:int,chapter_title:str,beat:str) -> str:
        bible=book['bible']
        return (
            'Escreve APENAS o texto de um capítulo ORIGINAL de um romance em português europeu natural. '
            'Usa ação, diálogos e descrição, com cenas e parágrafos; evita esquemas, comentários técnicos e resumos. '
            'Mantém coerência de nomes, acontecimentos e objetivos. Não repitas capítulos anteriores. '
            'Não apresentes um final definitivo antes do último capítulo. '\
            f'Tema: {book["theme"]}. Título da obra: {book["title"]}. '
            f'Protagonista: {bible["protagonist"]}. Cenário: {bible["setting"]}. '
            f'Descoberta central: {bible["discovery"]}. Conflito central: {bible["conflict"]}. '
            f'Capítulo {number}/{book["planned"]}: {chapter_title}. Objetivo narrativo: {beat} '
            f'Continuidade: {self._context(book,number)}\n'
            'Escreve agora o capítulo em prosa original, com progressão dramática. '
            'Usa um final de capítulo que se liga naturalmente ao seguinte.'
        )

    @staticmethod
    def _offline_chapter(book:dict,number:int,title:str,beat:str) -> str:
        """Non-neural, bounded fallback with chapter-specific continuity."""
        b=book['bible']; r=random.Random(book['seed']+number*7919)
        person=b['protagonist'].split(',')[0]
        intrigue=r.choice((
            'um ruído atrás da parede interrompeu a conversa',
            'uma anotação antiga revelou um detalhe que ninguém tinha visto',
            'uma testemunha contradisse a versão mais confortável',
            'um objeto desapareceu precisamente quando mais precisavam dele',
        ))
        helper=r.choice(('a vizinha','o vigia','uma antiga amiga','o aprendiz da oficina'))
        outcome=r.choice(('uma fotografia manchada','um nome riscado num caderno',
                          'uma marca quase invisível','a memória de uma promessa'))
        previous='Desde o capítulo anterior, as decisões tinham consequências.' if number>1 else (
            f'Tudo começou {b["setting"]}, onde {person} vivia sem imaginar a mudança que se aproximava.')
        close=('O conflito não estava resolvido. Pelo contrário: a próxima decisão parecia ainda mais difícil.'
               if number<book['planned'] else
               'Desta vez, a escolha tinha consequências reais. O mistério deixava de controlar a vida de todos; era o começo de uma vida diferente.')
        return '\n\n'.join([
            f'{previous} Àquela hora, a luz desenhava linhas incertas pelas janelas. {person} demorou-se junto à porta, a pensar em {b["theme"]}. Havia coisas que uma pessoa podia ignorar durante anos, até ao momento em que deixavam de caber no silêncio.',
            f'Foi então que {intrigue}. {person} segurou a respiração e voltou a examinar {b["discovery"]}. Parecia a mesma coisa de sempre, mas uma nova pergunta mudava tudo. Talvez a resposta estivesse menos no objeto do que na forma como as pessoas o observavam.',
            f'— Não podes continuar a fazer isto sozinho — disse {helper}.\n— Então ajuda-me a perceber o que estamos a perder — respondeu {person}.\nA conversa não resolveu o problema, mas transformou uma suspeita numa hipótese que ambos podiam discutir. E isso já era um avanço.',
            f'{beat} Na prática, significava abandonar uma certeza antiga e experimentar outra abordagem. {person} registou o que sabia, separou o que apenas imaginava e decidiu começar pela prova mais simples. Se falhasse, pelo menos saberia que caminho evitar na próxima tentativa.',
            f'A primeira experiência trouxe {outcome}. O pequeno detalhe parecia afastado do mistério principal, mas depressa se tornou impossível ignorá-lo. Era uma pista com duas leituras opostas. O próprio {person} teve de admitir que podia estar enganado — e que a verdade não era menos importante por isso.',
            f'Quando a noite chegou, {b["conflict"]}. As pessoas do lugar esperavam uma resposta rápida; {person} só tinha perguntas melhores. Ainda assim, escolher não era o mesmo que ter a certeza. Fez uma promessa a si próprio: o passo seguinte teria de proteger quem podia ser magoado.',
            f'{close} Antes de apagar a luz, {person} voltou ao caderno e escreveu uma única frase: «Talvez a resposta esteja na pergunta que ainda não tive coragem de fazer.»'
        ])

    def continue_book(self,language,amount:int=1) -> dict:
        if not 1<=amount<=3:raise ValueError('Podes escrever entre 1 e 3 capítulos de cada vez.')
        book=self.active()
        if book is None:return {'error':'Ainda não existe nenhum livro. Diz «cria um livro sobre ...».'}
        written=[]
        for number in range(book['written']+1,min(book['planned'],book['written']+amount)+1):
            chapter_title,beat=_chapter_outline(book['planned'],number)
            prompt=self._prompt(book,number,chapter_title,beat)
            tokens=int(os.getenv('NORD_STORY_CHAPTER_TOKENS','650'))
            tokens=min(1050,max(200,tokens))
            output=language.generate(prompt,creative=True,max_tokens=tokens,remember=False) if language else None
            if output is not None and len(output.strip())<180:output=None
            source='Transformer Nord nativo (experimental)' if output else 'gerador procedural sem fluência neuronal'
            body=output.strip() if output else self._offline_chapter(book,number,chapter_title,beat)
            recap=(f'{chapter_title}: {beat} '
                   f'Início: {body[:170].strip()} Final: {body[-240:].strip()}')[:550]
            # Save each completed chapter independently; never overwrite a
            # previous chapter after a crash or model timeout.
            with self.db:
                self.db.execute('INSERT INTO chapters(book_id,number,title,body,recap,created) VALUES(?,?,?,?,?,?)',
                                (book['id'],number,chapter_title,body,recap,int(time.time())))
            written.append({'number':number,'title':chapter_title,'body':body,'source':source})
        return {'book':self.active(),'chapters':written}

    def revise_last(self, language, *, kind: str = 'longer') -> dict | None:
        """Revise the last saved chapter, preserving its book and chapter ID."""
        book = self.active()
        if not book or not book['written']:
            return None
        number = book['written']
        row = self.db.execute('SELECT title,body FROM chapters WHERE book_id=? AND number=?',
                              (book['id'],number)).fetchone()
        if row is None:
            return None
        title, original = row
        direction = {
            'longer':'Reescreve o capítulo por completo e torna-o substancialmente mais longo.',
            'rewrite':'Reescreve o capítulo sem alterar os factos centrais.',
            'ending':'Cria um final alternativo para este capítulo, preservando as personagens.'
        }.get(kind,'Reescreve com melhor desenvolvimento.')
        prompt = (direction + ' Mantém a continuidade e o português europeu. '
                  'Responde apenas com a prosa revista.\n'
                  'LIVRO: '+book['title']+'\nCAPÍTULO ANTERIOR:\n'+original[:5000])
        output = language.generate(prompt,creative=True,max_tokens=1200,
                                   remember=False) if language else None
        if not output or (kind == 'longer' and len(output)<len(original)*1.2):
            from .story_revision import revise_story
            output = revise_story(original,kind)
        recap = (f'{title}: Início: {output[:170].strip()} '
                 f'Final: {output[-240:].strip()}')[:550]
        with self.db:
            self.db.execute('UPDATE chapters SET body=?,recap=? WHERE book_id=? AND number=?',
                            (output,recap,book['id'],number))
        return {'number':number,'title':title,'body':output}

    def export(self) -> Path | None:
        book=self.active()
        if not book:return None
        rows=self.db.execute('SELECT number,title,body FROM chapters WHERE book_id=? ORDER BY number',
                             (book['id'],)).fetchall()
        output=self.root/'books'/f'livro-{book["id"]:03d}.md'
        output.parent.mkdir(parents=True,exist_ok=True)
        intro=(f'# {book["title"]}\n\nTema: {book["theme"]}\n\n'
               f'*{len(rows)} de {book["planned"]} capítulos escritos.*\n\n')
        content=intro+'\n\n'.join(f'## Capítulo {n} — {title}\n\n{body}' for n,title,body in rows)+'\n'
        fd,tmp=tempfile.mkstemp(prefix='.livro-',suffix='.md',dir=output.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as stream:
                stream.write(content)
                stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,output)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        return output

    def close(self):
        self.db.close()
