"""Human-in-the-loop controls for hypothesis-driven training.

The teacher can pause autonomous updates and require approval of mathematics
hypotheses. Verified arithmetic is NOT replaced by an unsupported correction.
Blocking future training cannot undo gradient steps already applied.
"""
from __future__ import annotations

import re
import unicodedata

from .hypotheses import verified_value
from .natural_math import read_math_question


def _normal(text: str) -> str:
    val = unicodedata.normalize('NFKD', text.strip().casefold())
    return ''.join(c for c in val if not unicodedata.combining(c))


def normalize_theory_question(text: str) -> str | None:
    solved = read_math_question(text.strip().strip('"“”'))
    if not solved:
        return None
    expr, _ = solved
    match = re.fullmatch(r'\s*(\d{1,3})\s*([+*×-])\s*(\d{1,3})\s*', expr)
    if not match:
        return None
    a, op, b = match.groups()
    return f'Quanto é {int(a)} {op.replace("*", "×")} {int(b)}?'


def route_control(brain, utterance: str) -> tuple[str, str] | None:
    """Return response when a spoken instruction controls *what* may train."""
    raw = utterance.strip().rstrip('!.? ')
    text = _normal(raw)
    mem = brain.memory
    if text in ('pausa aprendizagem','pausa o treino','para de aprender sozinho',
                'para de treinar','pausa as experiencias','pausa experiencias'):
        mem.pause_autonomy(True)
        return ('Parei o treino autónomo, incluindo experiências no mundo virtual e pesquisa autónoma. '
                'Ainda podes ensinar-me diretamente. Escreve «retoma aprendizagem» para continuar.'), 'controlo humano'
    if text in ('retoma aprendizagem','retoma o treino','continua a aprender',
                'volta a aprender','retoma experiencias'):
        mem.pause_autonomy(False)
        return ('Reativei as experiências e o treino autónomo. '
                'Vou respeitar as hipóteses bloqueadas e as que aguardam aprovação.'), 'controlo humano'
    if text in ('quero aprovar as hipoteses','quero aprovar hipoteses',
                'pede me aprovacao para aprender','revisa as hipoteses comigo',
                'modo de revisao','modo revisao'):
        mem.set_review_mode(True)
        return ('Modo de revisão ligado. As hipóteses matemáticas aguardam a tua aprovação '
                'antes de entrarem no treino. Podes dizer «mostra hipóteses», '
                '«aprova hipótese 3+2» ou «rejeita hipótese 3+2». '
                'Outros treinos autónomos continuam ativos; para os parar, diz «pausa aprendizagem».\n'
                'Não consigo remover automaticamente dos pesos aquilo que já foi treinado.'), 'controlo humano'
    if text in ('aprender hipoteses automaticamente','aprova automaticamente as hipoteses',
                'desliga modo revisao','desliga modo de revisao'):
        mem.set_review_mode(False)
        return ('Desativei a aprovação individual das hipóteses matemáticas. '
                'Continuarei a verificá-las antes do treino.'), 'controlo humano'
    if text in ('mostra hipoteses','mostra as hipoteses','que teorias tens',
                'mostra teorias','mostra as teorias','o que estas a experimentar'):
        rows=mem.theory_details(limit=10)
        if not rows:
            return 'Ainda não tenho hipóteses matemáticas na fila. Ensina-me uma conta como «2+2 é 4».','laboratório de teorias'
        lines=['Teorias e experiências recentes:']
        for q,a,state,why,pred in rows:
            detail=f' | primeira previsão: {pred}' if pred else ''
            lines.append(f'  • {q} = {a} | {state} | {why}{detail}')
        lines.append('Para controlar: «aprova hipótese 3+2», «rejeita hipótese 3+2», «pausa aprendizagem».')
        return '\n'.join(lines), 'laboratório de teorias'
    command=re.fullmatch(r'(aprova|rejeita)\s+(?:a\s+)?hip[oó]tese\s+(.+)',raw,re.I)
    if command:
        action,expr=command.groups()
        question=normalize_theory_question(expr)
        if not question:
            return 'Não consegui identificar a hipótese. Exemplo: «aprova hipótese 3+2».', 'controlo humano'
        if action.casefold()=='aprova':
            ok=mem.approve_theory(question)
            return (f'Aprovada: {question}. Será verificada pelo cálculo antes do treino.' if ok
                    else 'Essa hipótese não está à espera de aprovação ou já foi treinada.'), 'controlo humano'
        else:
            mem.block_training(question)
            return (f'Bloqueei {question} para novos treinos e revisões. '
                    'Se os pesos já foram atualizados, o bloqueio não desfaz esse treino anterior.'), 'controlo humano'
    return None
