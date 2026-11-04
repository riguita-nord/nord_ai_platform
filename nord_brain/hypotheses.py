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


def explore_theories(expression: str, answer: str, limit: int = 40) -> list[tuple[str, str, str]]:
    """Explore changes to the input and their predicted consequences.

    A verified seed *starts* a search, not a table of the same answer. Each
    proposal includes a human-readable counterfactual explanation. Labels are
    calculated independently, and neither the teacher nor the neural model is
    trusted to invent arithmetic truth.
    """
    match = _PAIR.fullmatch(expression)
    if match is None or verified_value(expression) != str(answer):
        return []
    a, op, b = int(match[1]), match[2].replace('x', '*').replace('×', '*'), int(match[3])
    if not (0 <= a <= 30 and 0 <= b <= 30):
        return []
    candidates = []
    def add(x, operation, y, theory):
        if not (0 <= x <= 35 and 0 <= y <= 35):
            return
        expr = f'{x} {operation} {y}'
        outcome = verified_value(expr)
        if outcome is not None:
            candidates.append((f'Quanto é {expr}?', outcome, theory))
    # Test differences, rather than overfitting to one fixed answer.
    for amount in (1, 2):
        add(a+amount, op, b, f'e se acrescentar {amount} ao primeiro número?')
        add(a, op, b+amount, f'e se acrescentar {amount} ao segundo número?')
        add(a-amount, op, b, f'e se retirar {amount} ao primeiro número?')
        add(a, op, b-amount, f'e se retirar {amount} ao segundo número?')
    add(b, op, a, 'e se trocar a ordem dos números?')
    add(a+1, op, b-1, 'e se transferir 1 do segundo para o primeiro?')
    add(a-1, op, b+1, 'e se transferir 1 do primeiro para o segundo?')
    add(a+1, op, b+1, 'e se acrescentar 1 a ambos?')
    add(a*2, op, b*2, 'e se duplicar ambos os números?')
    for other in ('+', '-', '×'):
        if other != op:
            add(a,other,b,f'e se trocar a operação {op} por {other}?')
    # Mix in regular exercises of varied outputs, not just nearby values.
    for offset in range(3, 10):
        add(a+offset, '+', b, f'e se aumentar o primeiro número em {offset}?')
    distinct = []
    seen = set()
    for item in candidates:
        if item[0] not in seen and item[0] != f'Quanto é {a} {op} {b}?':
            seen.add(item[0])
            distinct.append(item)
    return distinct[:max(1, min(int(limit), 48))]
