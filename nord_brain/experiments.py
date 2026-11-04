"""Deterministic, automatically checkable curricula with disjoint held-out inputs.

The program checks answers by arithmetic rather than asking its own model to
mark its predictions. These narrow tasks do not prove general reasoning.
"""
from __future__ import annotations
from random import Random


def curriculum():
    train=[]
    tests=[]
    for a in range(10):
        for b in range(10):
            # Preserve the original addition curriculum and its held-out pairs.
            example=(f'Quanto é {a} + {b}?',str(a+b))
            (tests if a+b==9 else train).append(example)
            # Compose operators: target is externally computed, never model-generated.
            for c in (2,3):
                chain=(f'Quanto é {a} + {b} × {c}?',str(a+b*c))
                grouped=(f'Quanto é ({a} + {b}) × {c}?',str((a+b)*c))
                # Holdout split uses inputs, not answers, so unseen combinations
                # are not accidentally included in gradient updates.
                target=tests if (a*7+b*3+c)%11==0 else train
                target.extend((chain,grouped))
    Random(2026).shuffle(train)
    return train,tests
