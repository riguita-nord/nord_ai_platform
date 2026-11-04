"""Small, explicit Portuguese dialogue parser.

This layer recognizes a few *teaching conventions*, not natural-language
understanding. Unknown input must not silently become a training label.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class Intent:
    kind: str
    question: str = ''
    answer: str = ''
    topic: str = ''


def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().strip('“”"'))



def normalize_request(text: str) -> str:
    """Normalise short spoken requests for stable alias matching."""
    ascii_text = unicodedata.normalize('NFKD', clean(text).casefold())
    ascii_text = ''.join(c for c in ascii_text if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', ascii_text)).strip()


def is_self_knowledge_request(text: str) -> bool:
    return normalize_request(text) in {
        'o que sabes', 'o que e que sabes', 'o que sabes fazer',
        'o que consegues fazer', 'o que aprendeste', 'o que ja aprendeste',
        'o que tens aprendido', 'quais sao as tuas capacidades',
        'que capacidades tens', 'quais sao as tuas competencias',
        'o que ja sabes fazer', 'diz me o que sabes',
        'diz me o que sabes fazer', 'diz me o que aprendeste',
        'mostra o que aprendeste', 'o que aprendeste ate agora',
        'que coisas aprendeste', 'o que sabes e o que aprendeste',
        'o que sabes fazer e o que aprendeste',
    }


def _means_self_knowledge(description: str) -> bool:
    meaning = normalize_request(description)
    markers = (
        'o que sabes fazer', 'o que consegues fazer',
        'o que aprendeste', 'o que ja aprendeste',
        'o que tens aprendido', 'as tuas capacidades',
        'as tuas competencias', 'o que sabes e',
        'o que consegues e', 'o teu progresso',
    )
    return any(marker in meaning for marker in markers)

def _topic(text: str) -> str:
    text = clean(text).rstrip('.?! ')
    return re.sub(r'^(?:o|a|os|as|um|uma)\s+', '', text, flags=re.I).strip()


def interpret(message: str, last_question: str | None = None, awaiting_explanation: bool = False) -> Intent:
    text = clean(message)
    if not text:
        return Intent('empty')
    if re.fullmatch(r'(?:olá|ola|oi|bom dia|boa tarde|boa noite)[!. ]*', text, re.I):
        return Intent('greeting')
    if re.fullmatch(r'(?:adeus|até logo|tchau)[!. ]*', text, re.I):
        return Intent('farewell')

    # A meta-instruction is not an answer to the preceding question.
    meta = re.match(
        r'^(?:quando\s+(?:eu\s+)?(?:te\s+)?(?:digo|disser|pergunto|perguntar)\s+)'
        r'(.+?)\s*[,;:]?\s*(?:eu\s+)?quero\s+que\s+(.+?)\s*[.!]?|o que (?:já )?aprendeste|quanto (?:já )?aprendeste|mostra o teu progresso|qual é o teu progresso)\??$', text, re.I):
        return Intent('progress')
    m = re.match(r'^(?:quando (?:eu )?(?:te )?(?:perguntar|disser))\s+["“]?(.+?)["”]?\s*[,;:]\s*(?:responde|diz)\s+["“]?(.+?)["”]?[.!]?$', text, re.I)
    if m:
        return Intent('teach', clean(m[1]).rstrip('"”'), clean(m[2]).rstrip('"”'))
    m = re.match(r'^(?:ensina-te|aprende)\s+(?:isto\s*:\s*)?(.+?)\s*=>\s*(.+)$',text,re.I)
    if m:
        return Intent('teach',clean(m[1]),clean(m[2]))
    m = re.match(r'^(?:(?:não|nao|errado|isso está errado|isso esta errado)\s*[,!:.-]?\s*)?(?:a resposta (?:correta|certa) é|o correto é)\s+(.+)$',text,re.I)
    if m and last_question:
        return Intent('correction',last_question,clean(m[1]))
    m = re.match(r'^(?:não|nao|errado|isso está errado|isso esta errado)\s*[,!:.-]\s*(.+)$',text,re.I)
    if m and last_question:
        return Intent('correction',last_question,clean(m[1]))
    m = re.match(r'^(?:o meu nome é|chamo-me|podes chamar-me)\s+([\wÀ-ÿ-]{2,40})[.!]?$', text,re.I)
    if m:
        return Intent('identity','Como me chamo?',m[1])
    m = re.match(r'^(?:pesquisa|procura|investiga|podes pesquisar|podes investigar)\s+(?:sobre\s+)?(.+?)[.?!]?$', text,re.I)
    if m:
        return Intent('search',topic=clean(m[1]).rstrip('?.!'))
    if re.fullmatch(r'(?:como me chamo|qual é o meu nome)\??',text,re.I):
        return Intent('question','Como me chamo?')
    m = re.match(r'^o que é\s+(.+?)\??$',text,re.I)
    if m:
        return Intent('definition_question',topic=_topic(m[1]))
    m = re.match(r'^(?:lembra-te (?:de )?que|fica a saber que|aprende que)\s+(.+)$',text,re.I)
    if m:
        text=clean(m[1])
        return _fact(text) or Intent('statement',answer=text)
    if text.endswith('?'):
        return Intent('question',text)
    fact=_fact(text)
    if fact:
        return fact
    # An explicit explanatory response following "Podes explicar?" is teaching.
    if awaiting_explanation and last_question and not text.endswith('?') and len(text)>=5:
        return Intent('explanation',last_question,text)
    return Intent('question',text)


def _fact(text: str) -> Intent | None:
    m=re.match(r'^((?:(?:o|a|um|uma)\s+)?[\wÀ-ÿ-]+(?:[ \wÀ-ÿ-]{0,45})?)\s+(?:é|significa)\s+(.+?)[.!]?$',text,re.I)
    if not m:
        return None
    term=_topic(m[1])
    meaning=clean(m[2]).rstrip('.')
    if len(term)>55 or len(meaning)<2 or len(meaning)>350:
        return None
    return Intent('fact',f'O que é {term}?',f'{term} é {meaning}.',term)
,
        text, re.I,
    )
    if meta:
        trigger = clean(meta.group(1)).strip('“”" ,;:.?!')
        desired = clean(meta.group(2)).rstrip('.!')
        if 2 <= len(trigger) <= 100:
            return Intent('intent_instruction', question=trigger, answer=desired,
                          topic='self_knowledge' if _means_self_knowledge(desired) else 'unsupported')
    if is_self_knowledge_request(text):
        return Intent('self_knowledge')
    if re.search(r'^(?:como está o (?:teu )?cérebro|o que (?:já )?aprendeste|quanto (?:já )?aprendeste|mostra o teu progresso|qual é o teu progresso)\??$', text, re.I):
        return Intent('progress')
    m = re.match(r'^(?:quando (?:eu )?(?:te )?(?:perguntar|disser))\s+["“]?(.+?)["”]?\s*[,;:]\s*(?:responde|diz)\s+["“]?(.+?)["”]?[.!]?$', text, re.I)
    if m:
        return Intent('teach', clean(m[1]).rstrip('"”'), clean(m[2]).rstrip('"”'))
    m = re.match(r'^(?:ensina-te|aprende)\s+(?:isto\s*:\s*)?(.+?)\s*=>\s*(.+)$',text,re.I)
    if m:
        return Intent('teach',clean(m[1]),clean(m[2]))
    m = re.match(r'^(?:(?:não|nao|errado|isso está errado|isso esta errado)\s*[,!:.-]?\s*)?(?:a resposta (?:correta|certa) é|o correto é)\s+(.+)$',text,re.I)
    if m and last_question:
        return Intent('correction',last_question,clean(m[1]))
    m = re.match(r'^(?:não|nao|errado|isso está errado|isso esta errado)\s*[,!:.-]\s*(.+)$',text,re.I)
    if m and last_question:
        return Intent('correction',last_question,clean(m[1]))
    m = re.match(r'^(?:o meu nome é|chamo-me|podes chamar-me)\s+([\wÀ-ÿ-]{2,40})[.!]?$', text,re.I)
    if m:
        return Intent('identity','Como me chamo?',m[1])
    m = re.match(r'^(?:pesquisa|procura|investiga|podes pesquisar|podes investigar)\s+(?:sobre\s+)?(.+?)[.?!]?$', text,re.I)
    if m:
        return Intent('search',topic=clean(m[1]).rstrip('?.!'))
    if re.fullmatch(r'(?:como me chamo|qual é o meu nome)\??',text,re.I):
        return Intent('question','Como me chamo?')
    m = re.match(r'^o que é\s+(.+?)\??$',text,re.I)
    if m:
        return Intent('definition_question',topic=_topic(m[1]))
    m = re.match(r'^(?:lembra-te (?:de )?que|fica a saber que|aprende que)\s+(.+)$',text,re.I)
    if m:
        text=clean(m[1])
        return _fact(text) or Intent('statement',answer=text)
    if text.endswith('?'):
        return Intent('question',text)
    fact=_fact(text)
    if fact:
        return fact
    # An explicit explanatory response following "Podes explicar?" is teaching.
    if awaiting_explanation and last_question and not text.endswith('?') and len(text)>=5:
        return Intent('explanation',last_question,text)
    return Intent('question',text)


def _fact(text: str) -> Intent | None:
    m=re.match(r'^((?:(?:o|a|um|uma)\s+)?[\wÀ-ÿ-]+(?:[ \wÀ-ÿ-]{0,45})?)\s+(?:é|significa)\s+(.+?)[.!]?$',text,re.I)
    if not m:
        return None
    term=_topic(m[1])
    meaning=clean(m[2]).rstrip('.')
    if len(term)>55 or len(meaning)<2 or len(meaning)>350:
        return None
    return Intent('fact',f'O que é {term}?',f'{term} é {meaning}.',term)
