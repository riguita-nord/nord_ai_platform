"""Safe extraction of simple arithmetic from ordinary Portuguese dialogue.

A deterministic arithmetic verifier is *not* neural reasoning. It provides
correct checked examples for training, without trusting arbitrary model output.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .reasoning import Reasoning, calculate

_WORD_NUMBERS = {
    'zero': '0', 'um': '1', 'uma': '1', 'dois': '2', 'duas': '2',
    'três': '3', 'tres': '3', 'quatro': '4', 'cinco': '5',
    'seis': '6', 'sete': '7', 'oito': '8', 'nove': '9',
    'dez': '10', 'onze': '11', 'doze': '12', 'treze': '13',
    'catorze': '14', 'quatorze': '14', 'quinze': '15',
    'dezasseis': '16', 'dezassete': '17', 'dezoito': '18',
    'dezanove': '19', 'vinte': '20',
}
_OPERATORS = {'mais': '+', 'menos': '-', 'vezes': '*', 'multiplicado por': '*',
              'a dividir por': '/', 'dividido por': '/'}

_PREFIX = re.compile(
    r'^(?:(?:e\s+)?(?:diz(?:[- ]me)?|podes\s+dizer(?:[- ]me)?)\s+)?'
    r'(?:quanto\s+(?:é|e|dá|da)(?:\s+que\s+(?:é|e|dá|da))?'
    r'|qual\s+(?:é|e)\s+o\s+resultado\s+de'
    r'|calcula|calcular|resolve)\s+', re.I,
)

_DEMONSTRATION = re.compile(
    r'(?<![\w])(?P<expr>-?\d{1,7}(?:\s*[+*/×÷−-]\s*-?\d{1,7}){1,6})'
    r'\s*(?:=|é|e|dá|da|igual\s+a)\s*'
    r'(?P<claimed>-?\d{1,9})(?!\d)', re.I,
)


def _as_expression(text: str) -> str | None:
    value = text.strip().rstrip('?!').strip()
    if value.endswith('.') and not re.search(r'\d\.\d$', value):
        value = value[:-1].strip()
    for words, sign in sorted(_OPERATORS.items(), key=lambda x: -len(x[0])):
        value = re.sub(r'\b' + re.escape(words) + r'\b', sign, value, flags=re.I)
    for word, number in _WORD_NUMBERS.items():
        value = re.sub(r'\b' + re.escape(word) + r'\b', number, value, flags=re.I)
    if not re.fullmatch(r'[0-9\s()+*/×÷−-]+', value):
        return None
    if len(value) > 75 or not re.search(r'\d\s*[+*/×÷−-]\s*\d', value):
        return None
    return value.replace('×', '*').replace('÷', '/').replace('−', '-')


def read_math_question(message: str) -> tuple[str, Reasoning] | None:
    """Only a whole arithmetic request is considered a math question."""
    stripped = ' '.join(message.strip().split())
    if len(stripped) > 150:
        return None
    matched = _PREFIX.match(stripped)
    expression = _as_expression(stripped[matched.end():] if matched else stripped)
    if expression is None:
        return None
    solution = calculate(expression)
    if solution is None:
        return None
    return expression, solution


@dataclass(frozen=True)
class VerifiedExample:
    expression: str
    claimed: int
    actual: str
    verified: bool
    proof: Reasoning


def read_math_example(message: str) -> VerifiedExample | None:
    """Read and check a small explicitly stated equation inside an explanation.

    Text around the example is not stored as an answer. This avoids training
    the entire instruction 'isto é matemática ...' as if it were a label.
    """
    if len(message) > 450:
        return None
    match = _DEMONSTRATION.search(message)
    if not match:
        return None
    expr = match.group('expr').replace('×', '*').replace('÷', '/').replace('−', '-')
    proof = calculate(expr)
    if proof is None:
        return None
    actual = proof.steps[-1].split(' = ')[-1]
    claimed = int(match.group('claimed'))
    return VerifiedExample(expr, claimed, actual, actual == str(claimed), proof)
