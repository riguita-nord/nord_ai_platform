"""Conservative offline quality gate for native neural dialogue.

A filter can reject obvious corruption; it cannot create fluency.
"""
from __future__ import annotations

import re

COMMON_SHORT = frozenset({
    'olá', 'ola', 'oi', 'boas', 'sim', 'não', 'claro', 'bem',
    'obrigado', 'obrigada', 'estou', 'aqui', 'tudo', 'contigo',
    'comigo', 'podes', 'posso', 'vamos', 'falar', 'conversar',
    'sobre', 'quero', 'queres', 'como', 'estás', 'hoje', 'bom',
    'dia', 'boa', 'tarde', 'noite', 'percebo', 'entendo', 'certo',
    'isso', 'é', 'uma', 'ideia', 'qual', 'o', 'a', 'de', 'e', 'tu',
})


def score_candidate(candidate, prompt):
    from .native_language import coherent_sample

    value = ' '.join(str(candidate).split())
    if not value or len(value) > 440:
        return None
    if re.search(r'[\x00-\x1f\ufffd]', value):
        return None
    if re.search(r'(?:pergunta:|resposta:|<\\|end\\|>|<unk>|###)', value, re.I):
        return None
    words = re.findall(r'\w+', value.casefold(), re.UNICODE)
    if not words:
        return None
    if sum(ch.isalpha() for ch in value) / max(1, len(value)) < .55:
        return None
    if len(words) >= 6:
        if not coherent_sample(value):
            return None
    else:
        # Do not mistake numeric debris and fragments for short replies.
        if any(ch.isdigit() for ch in value):
            return None
        if len(set(words)) < max(1, len(words) - 1):
            return None
        if len(words) > 1 and sum(len(w) == 1 for w in words) > 1:
            return None
        if not any(word in COMMON_SHORT for word in words):
            return None
        if len(words) <= 3 and len(words) > 1 and any(len(w) == 1 for w in words):
            return None
    topic = set(re.findall(r'\w{4,}', prompt.casefold()))
    overlap = len(set(words) & topic)
    repetition = len(words) - len(set(words))
    return overlap * 2 - repetition - max(0, len(words) - 45) * .1


def choose_candidate(candidates, prompt):
    ranked = [(score_candidate(value, prompt), value) for value in candidates]
    valid = [(score, value) for score, value in ranked if score is not None]
    return max(valid, key=lambda row: row[0])[1] if valid else None
