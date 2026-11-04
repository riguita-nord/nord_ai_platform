"""Small deterministic, verifiable experiences: not general intelligence.

Training/holdout questions use disjoint input pairs. No language model is used
to invent answers or mark its own work correct.
"""
from __future__ import annotations
from random import Random


def curriculum():
    held = {(0,9),(1,8),(2,7),(3,6),(4,5),(5,4),(6,3),(7,2),(8,1),(9,0)}
    tests = [(f'Quanto é {a} + {b}?',str(a+b)) for a,b in sorted(held)]
    training = [(f'Quanto é {a} + {b}?',str(a+b))
                for a in range(10) for b in range(10) if (a,b) not in held]
    Random(2026).shuffle(training)
    return training, tests
