"""Verified mathematical hypothesis generation from teacher experiences.

A teacher example such as 2+2=4 is a seed, not a table to copy. Candidate
problems are generated from a bounded arithmetic search, predicted by the
Transformer BEFORE verification, and only correct targets are trained.
The mathematical oracle checks claims; it is not proof of neural competence.
"""
from __future__ import annotations

from random import Random
import re

from .reasoning import calculate


_PAIR = re.compile(r'^\s*(\d{1,3})\s*([+*x×-])\s*(\d{1,3})\s*$')


def verified_value(expression: str) -> str | None:
    result = calculate(expression)
    if result is None or not result.steps:
        return None
    return result.steps[-1].split(' = ')[-1]


def propose(expression: str, answer: str, limit: int = 32) -> list[tuple[str, str]]:
    """Generate distinct *testable* hypotheses, not trusted answers.

    We explore alternative sums giving the seed's output, subtraction and
    multiplication. Correctness is checked independently by the verifier.
    """
    match = _PAIR.fullmatch(expression)
    if not match or verified_value(expression) != str(answer):
        return []
    total = int(answer)
    if not 0 <= total <= 30:
        return []
    candidates = []
    for a in range(total + 1):
        candidates.append((f'{a} + {total - a}', str(total)))
    for offset in range(0, min(9, 31-total)):
        candidates.append((f'{total+offset} - {offset}', str(total)))
    for a in range(1, total+1):
        if total % a == 0 and total // a <= 30:
            candidates.append((f'{a} × {total//a}', str(total)))
    # Include a few contrasts with DIFFERENT outputs so the neural model
    # cannot reduce every input to the one seeded answer.
    for a in range(min(total, 8)+1):
        candidates.append((f'{a} + {total + 1 - a}', str(total+1)))
    for a in range(min(total, 8)+1):
        if total > 0:
            candidates.append((f'{a} + {total - 1 - a}', str(total-1)))
    rng = Random(total * 2741 + 2026)
    rng.shuffle(candidates)
    unique = []
    seen = set()
    for expression,claimed in candidates:
        q = f'Quanto é {expression}?'
        if q in seen:
            continue
        seen.add(q)
        # Every label must come from the verifier, never the neural guess.
        checked = verified_value(expression)
        if checked is None or checked != claimed:
            continue
        unique.append((q,checked))
        if len(unique) >= max(1,min(limit,48)):
            break
    return unique
