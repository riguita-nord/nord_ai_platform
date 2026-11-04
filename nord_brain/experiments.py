"""Procedural arithmetic practice with genuinely unseen evaluations.

The verifier calculates targets for generated experiences; it does NOT teach
answers by encoding a list of specific user questions. For any claimed neural
ability, evaluate the Transformer on held-out expressions, not this verifier.
"""
from __future__ import annotations
from random import Random


MAX_LEVEL = 3


def _examples(level: int):
    """Produce progressively broader tasks, not literal memorised Q/A pairs."""
    if level == 0:
        # Single-digit addition, every ordered pair (including unseen examples).
        for a in range(10):
            for b in range(10):
                yield a, b, '+'
    elif level == 1:
        # Generalise addition to two digits and introduce subtraction.
        for a in range(20):
            for b in range(20):
                yield a, b, '+'
                if a >= b:
                    yield a, b, '-'
    elif level == 2:
        rng = Random(202602)
        for _ in range(1400):
            a, b = rng.randrange(100), rng.randrange(100)
            op = '+' if rng.randrange(2) else '-'
            if op == '-' and a < b:
                a, b = b, a
            yield a, b, op
    else:
        rng = Random(202603)
        for _ in range(1400):
            op = ('+', '-', '×')[rng.randrange(3)]
            a, b = rng.randrange(100), rng.randrange(100)
            if op == '×':
                a, b = rng.randrange(20), rng.randrange(20)
            if op == '-' and a < b:
                a, b = b, a
            yield a, b, op


def curriculum(level: int = 0):
    """Disjoint reproducible train/evaluation expressions for a skill level.

    No evaluation input is generated into the training partition. Repeating
    `curriculum` yields the same partitions across restarts. A teacher's
    existing stored example may be excluded from the evaluation at runtime.
    """
    if level not in range(MAX_LEVEL + 1):
        raise ValueError('Nível de matemática não suportado')
    training, tests = [], []
    seen = set()
    op_ids = {'+': 0, '-': 1, '×': 2}
    for a, b, op in _examples(level):
        q = f'Quanto é {a} {op} {b}?'
        if q in seen:
            continue
        seen.add(q)
        result = a+b if op == '+' else a-b if op == '-' else a*b
        example = q, str(result)
        # Stable partition based on inputs/operator, NOT on their answer.
        # Unlike a memorised sum table, the full input space extends by level.
        if (a*7 + b*13 + op_ids[op]*5) % 9 == 8:
            tests.append(example)
        else:
            training.append(example)
    Random(2026 + level).shuffle(training)
    Random(1526 + level).shuffle(tests)
    return training, tests
