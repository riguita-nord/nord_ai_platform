"""Offline ranking of native language candidates."""
from __future__ import annotations
import re

def score_candidate(candidate, prompt):
    from .native_language import coherent_sample
    if not coherent_sample(candidate):
        return None
    words=re.findall(r'\w+',candidate.casefold())
    topic=set(re.findall(r'\w{4,}',prompt.casefold()))
    overlap=len(set(words)&topic)
    repetition=len(words)-len(set(words))
    return overlap*2-repetition-max(0,len(words)-45)*.1

def choose_candidate(candidates,prompt):
    ranked=[(score_candidate(value,prompt),value) for value in candidates]
    valid=[(score,value) for score,value in ranked if score is not None]
    return max(valid,key=lambda row:row[0])[1] if valid else None
