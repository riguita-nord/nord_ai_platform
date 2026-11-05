"""Offline ranking of native language candidates.

Filters obvious corruption without demanding long, scripted-looking replies.
The score is a presentation heuristic, not a claim of semantic understanding.
"""
from __future__ import annotations
import re


def score_candidate(candidate, prompt):
    from .native_language import coherent_sample
    value = ' '.join(str(candidate).split())
    if not value or len(value) > 440:
        return None
    if re.search(r'[\x00-\x1f\ufffd]', value):
        return None
    if re.search(r'(?:pergunta:|resposta:|<\|end\|>|<unk>|###)', value, re.I):
        return None
    words = re.findall(r'\w+', value.casefold(), re.UNICODE)
    if not words or sum(ch.isalpha() for ch in value) / max(1, len(value)) < .55:
        return None
    # Short answers are legitimate. Previously ALL replies under six words
    # were rejected, including natural greetings and direct acknowledgements.
    if len(words) >= 6 and not coherent_sample(value):
        return None
    if len(words) <= 5:
        if len(set(words)) < max(1, len(words) - 1):
            return None
        if len(words) > 1 and sum(len(w) == 1 for w in words) > 1:
            return None
        # A tiny byte-level model frequently emits isolated numbers and
        # plausible-looking fragments (e.g. "e diz1." or "quanto diz-me 2.").
        # Do not treat them as successful conversation.
        if re.search(r'\\d', value):
            return None
        if re.search(r'(?i)\\b(?:diz-me|quanto|que|como)\\s*
    topic = set(re.findall(r'\w{4,}', prompt.casefold()))
    overlap = len(set(words) & topic)
    repetition = len(words) - len(set(words))
    return overlap * 2 - repetition - max(0, len(words) - 45) * .1


def choose_candidate(candidates, prompt):
    ranked = [(score_candidate(value, prompt), value) for value in candidates]
    valid = [(score, value) for score, value in ranked if score is not None]
    return max(valid, key=lambda row: row[0])[1] if valid else None
, value):
            return None
        from .native_language import coherent_sample
        # Short replies cannot pass the longer-sentence heuristic; instead
        # require recognizable PT-PT conversational words, not random syllables.
        common = {'olá','oi','boas','sim','não','claro','bem','obrigado',
                  'obrigada','estou','aqui','tudo','contigo','comigo',
                  'podes','posso','vamos','falar','conversar','sobre',
                  'quero','queres','como','estás','hoje','bom','dia',
                  'boa','tarde','noite','percebo','entendo','certo',
                  'isso','é','uma','ideia','qual','o','a','de','e','tu'}
        if not any(word in common for word in words):
            return None
        # Reject single-character-heavy fragments; a real short greeting
        # can be a single recognized word.
        if len(words) >= 2 and sum(len(w) == 1 for w in words) > 0 and len(words) <= 3:
            return None
    topic = set(re.findall(r'\w{4,}', prompt.casefold()))
    overlap = len(set(words) & topic)
    repetition = len(words) - len(set(words))
    return overlap * 2 - repetition - max(0, len(words) - 45) * .1


def choose_candidate(candidates, prompt):
    ranked = [(score_candidate(value, prompt), value) for value in candidates]
    valid = [(score, value) for score, value in ranked if score is not None]
    return max(valid, key=lambda row: row[0])[1] if valid else None
