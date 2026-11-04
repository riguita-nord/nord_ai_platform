"""Conservative Portuguese teaching and dialogue intent parser.

Recognizes a small set of forms; this is NOT neural natural-language understanding.
Meta-instructions are classified before free-form explanations so that directions
about the assistant's behaviour are never memorised as a literal answer.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import unicodedata

from .natural_math import read_math_question, read_math_example


@dataclass(frozen=True)
class Intent:
    kind: str
    question: str = ''
    answer: str = ''
    topic: str = ''


def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text).strip().strip('“”"'))


def normalize_request(text: str) -> str:
    """Normalise a short spoken request for stable, safe intent alias matching."""
    ascii_text = unicodedata.normalize('NFKD', clean(text).casefold())
    ascii_text = ''.join(c for c in ascii_text if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', ascii_text)).strip()


def is_self_knowledge_request(text: str) -> bool:
    request = normalize_request(text)
    return request in {
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

    # Parse a spoken intent correction first, even when the brain just asked
    # "Podes explicar-me?". The text is a *rule*, not the answer to that question.
    meta = re.match(
        r'^(?:quando\s+(?:eu\s+)?(?:te\s+)?(?:digo|disser|pergunto|perguntar)\s+)'
        r'(.+?)\s*[,;:]?\s*(?:eu\s+)?quero\s+que\s+(.+?)\s*[.!]?$',
        text, re.I,
    )
    if meta:
        trigger = clean(meta.group(1)).strip('“”" ,;:.?!')
        desired = clean(meta.group(2)).rstrip('.!')
        if 2 <= len(trigger) <= 100:
            return Intent('intent_instruction', question=trigger, answer=desired,
                          topic='self_knowledge' if _means_self_knowledge(desired) else 'unsupported')

    # Whole arithmetic queries and verified worked examples take priority over
    # definitions and over the previous question's answer slot.
    if read_math_question(text) is not None:
        return Intent('math_question', question=text)
    if read_math_example(text) is not None:
        return Intent('math_example', question=text)
    if normalize_request(text) in ('nao sei', 'agora nao consigo', 'nao consigo',
                                   'nao tenho a certeza', 'nao faco ideia'):
        return Intent('uncertain')

    if is_self_knowledge_request(text):
        return Intent('self_knowledge')

    if re.search(r'^(?:como está o (?:teu )?cérebro|quanto (?:já )?aprendeste|mostra o teu progresso|qual é o teu progresso)\??$', text, re.I):
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
    if awaiting_explanation and last_question and not text.endswith('?'):
        # Long, unstructured feedback is not automatically a correct answer.
        # Ask for a clear response rather than poisoning long-term memory.
        if 1 <= len(text) <= 110 and not re.search(
            r'\b(?:quero que|quando te|quando eu|seria o mesmo|se te disser|nao sei|não sei)\b',
            text, re.I,
        ):
            return Intent('explanation',last_question,text)
        return Intent('clarify_teaching',question=last_question)
    return Intent('question',text)


def _fact(text: str) -> Intent | None:
    # A command or question containing 'é' is NOT a definition. In particular,
    # 'e diz-me quanto é 2+2' must not become a concept named 'e diz-me quanto'.
    if re.match(r'^(?:(?:e\s+)?(?:diz(?:[- ]me)?|quanto|qual|podes|como|quem|explica|porque|por\s+que|isto)\b)', text, re.I):
        return None
    if re.search(r'\d\s*[+*/×÷−-]\s*\d', text) or '?' in text:
        return None
    m=re.match(r'^((?:(?:o|a|um|uma)\s+)?[\wÀ-ÿ-]+(?:[ \wÀ-ÿ-]{0,45})?)\s+(?:é|significa)\s+(.+?)[.!]?$',text,re.I)
    if not m:
        return None
    term=_topic(m[1])
    meaning=clean(m[2]).rstrip('.')
    if len(term)>55 or len(meaning)<2 or len(meaning)>350:
        return None
    return Intent('fact',f'O que é {term}?',f'{term} é {meaning}.',term)
