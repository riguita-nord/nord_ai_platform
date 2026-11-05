"""Compositional dialogue-act detection for short social exchanges.

This layer only handles conversational protocol (greeting, answer, reciprocal
question). It is not a substitute for a trained generative language model.
Uncertain utterances abstain instead of being treated as factual questions.
"""
from __future__ import annotations
import re
import unicodedata

def _words(text: str) -> list[str]:
    value=unicodedata.normalize('NFKD',text.casefold())
    value=''.join(c for c in value if not unicodedata.combining(c))
    return re.findall(r'[a-z0-9]+',value)

def social_act(message: str, previous_reply: str = '') -> str | None:
    words=_words(message)
    if not words or len(words)>12:
        return None
    tokens=set(words)
    has_well=bool(tokens & {'bem','fixe','tranquilo','otimo','boa','boas'})
    first_person=bool(tokens & {'tou','to','estou','ando','sinto'})
    reciprocal=any(words[i:i+2] in (['e','tu'],['e','contigo']) for i in range(len(words)-1))
    reciprocal=reciprocal or ('tu' in tokens and 'como' in tokens)
    if reciprocal and (has_well or first_person or len(words)<=5):
        return 'reciprocal'
    if has_well and (first_person or 'obrigado' in tokens or 'obrigada' in tokens):
        return 'answer'
    if not has_well:
        return None
    if tokens - {'tudo','bem','ta','esta','está','sim','pois','ok','fixe','tranquilo',
                 'otimo','obrigado','obrigada','e','por','ai','contigo','com','tu'}:
        return None
    previous=_words(previous_reply)
    if '?' in previous_reply and (set(previous)&{'como','estas','estás','bem','contigo'}):
        return 'answer'
    if '?' in message or set(previous)&{'ola','boas','viva'}:
        return 'question'
    return 'acknowledge'

def social_response(act: str) -> str | None:
    return {
        'reciprocal': 'Ainda bem! Estou por aqui e pronto para conversar. Sobre o que te apetece falar?',
        'answer': 'Ainda bem! Queres contar-me alguma coisa sobre o teu dia?',
        'question': 'Estou por aqui e pronto para conversar. E contigo, tudo bem?',
        'acknowledge': 'Combinado. Queres continuar?',
    }.get(act)

def conversational_answer(message: str, previous_reply: str = '') -> str | None:
    words=_words(message)
    previous=_words(previous_reply)
    if not words or len(words)>12 or '?' in message:
        return None
    if not ('falar' in previous and ('sobre' in previous or 'apetece' in previous)):
        return None
    if words in (['sobre','tudo'],['tudo'],['qualquer','coisa'],['um','pouco','de','tudo'],['sobre','qualquer','coisa'],['sobre','um','pouco','de','tudo']):
        return 'Podemos falar de tudo um pouco! Preferes tecnologia, música ou outro assunto?'
    if words in (['nao','sei'],['tanto','faz'],['tu','escolhe']):
        return 'Então escolho eu: qual foi a coisa mais interessante do teu dia?'
    return None


def topic_request(message: str) -> bool:
    """Detect a request for the assistant to initiate a subject, not an answer."""
    words=_words(message)
    tokens=set(words)
    choose=bool(tokens & {'escolhe','escolher','sugere','sugerir','inventa','propoe'})
    topic=bool(tokens & {'tema','assunto','conversa','aleatorio','qualquer'})
    return (choose and topic) or words in (
        ['uma','coisa','qualquer'], ['tu','escolhe'], ['tanto','faz'])


def suggest_topic(dialogue) -> str:
    """Cycle across distinct open-ended subjects; no claim of neural understanding."""
    topics=(
        'Já pensaste como seria viver numa cidade construída no oceano? O que achas que mudaria no dia a dia?',
        'Vamos falar de tecnologia: que invenção gostavas que existisse daqui a dez anos?',
        'Tenho um tema: se pudesses aprender uma habilidade instantaneamente, qual escolherias?',
        'E se as pessoas pudessem explorar outros planetas como hoje viajam entre países? Gostavas de ir?',
        'Vamos imaginar uma coisa: como seria um videojogo em que cada personagem tivesse a sua própria história?',
    )
    previous=dialogue.latest()
    last=previous['reply'] if previous else ''
    # Avoid repeating the immediately preceding topic even after a restart.
    for item in topics:
        if item != last:
            return item
    return topics[0]
