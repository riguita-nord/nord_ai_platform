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


# Ambiguous literal forms (dates and numeric intervals) are not arithmetic
# unless the speaker explicitly requests a calculation. This also avoids
# treating dates such as 2026-11-4 as a series of subtractions.
_DATE = re.compile(r'^\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}$')
_RANGE = re.compile(r'^\d{1,5}\s*[-–]\s*\d{1,5}$')


def _as_expression(text: str, *, explicitly_requested: bool = False) -> str | None:
    value = text.strip().rstrip('?!').strip()
    if value.endswith('.') and not re.search(r'\d\.\d$', value):
        value = value[:-1].strip()
    if _DATE.fullmatch(value):
        return None
    if not explicitly_requested and _RANGE.fullmatch(value):
        # "2-3" could mean players 2 to 3, a range, or a subtraction.
        # Prefer asking for "quanto é 2-3" rather than deciding silently.
        return None
    if not explicitly_requested and re.fullmatch(r'\d{1,2}/\d{1,2}(?:/\d{2,4})?', value):
        return None
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
    """Accept maths only when the *entire* message is an arithmetic request.

    Arithmetic fragments within a conversation, version, date, or player count
    must never automatically become a calculator request.
    """
    stripped = ' '.join(message.strip().split())
    if len(stripped) > 150:
        return None
    matched = _PREFIX.match(stripped)
    # Follow-up "e 3+5" is mathematical, unlike a narrative beginning with 'e'.
    candidate = stripped[matched.end():] if matched else stripped
    if not matched and re.match(r'^e\s+', candidate, re.I):
        candidate = re.sub(r'^e\s+', '', candidate, flags=re.I)
    expression = _as_expression(candidate, explicitly_requested=bool(matched))
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


_MATH_FRAMING = re.compile(
    # Anchored teaching cues, not arbitrary mention of 'conta' or 'resultado'
    # somewhere in unrelated dialogue about an account or game server.
    r'^(?:isto\s+(?:[ée]\s+)?matem[aá]tica|'
    r'(?:o\s+)?resultado(?:\s+(?:correto|certo))?\s+(?:[ée]|de)|'
    r'(?:a\s+)?(?:conta|equa[cç][aã]o|c[aá]lculo)\s+(?:[ée]|de)|'
    r'(?:aprende|ensina-te|fica\s+a\s+saber)(?:\s+(?:que|esta\s+conta))?|'
    r'(?:um\s+)?exemplo\s+de\s+matem[aá]tica(?:\s+[ée])?|'
    r'vamos\s+calcular)\s*[,;:]?$', re.I,
)


def read_math_example(message: str) -> VerifiedExample | None:
    """Verify explicit math demonstrations, not incidental text with numbers.

    Previously ``_DEMONSTRATION.search`` accepted any arithmetic-looking
    fragment anywhere in a sentence. A player talking about code or game
    players could thus trigger math hypotheses and unwanted neural training.
    A stand-alone equation or a message *explicitly framed* as mathematical
    teaching is now required. The bounds still reject unsafe expressions.
    """
    if len(message) > 450:
        return None
    match = _DEMONSTRATION.search(message)
    if not match:
        return None
    prefix = message[:match.start()].strip(' \t\n\r,;:.!?"“”')
    suffix = message[match.end():].strip(' \t\n\r,;:.!?"“”')
    explicit_context = bool(_MATH_FRAMING.fullmatch(prefix))
    # A bare equation is allowed, as is math-specific teaching context.
    # Narrative suffixes ("tenho 2+2=4 carros") are not a demonstration.
    if (prefix or suffix) and not explicit_context:
        return None
    expr = match.group('expr').replace('×', '*').replace('÷', '/').replace('−', '-')
    # Date-like content must not become a worked subtraction even in a demo.
    if _DATE.fullmatch(expr):
        return None
    proof = calculate(expr)
    if proof is None:
        return None
    actual = proof.steps[-1].split(' = ')[-1]
    claimed = int(match.group('claimed'))
    return VerifiedExample(expr, claimed, actual, actual == str(claimed), proof)
