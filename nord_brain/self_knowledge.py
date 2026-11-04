"""Grounded self-report from actual runtime features and teacher memory.

The Transformer cannot introspect its weights. This report is assembled from
program capabilities and saved observations, not from neural self-awareness.
"""
from __future__ import annotations


def report(memory, core, *, internet_enabled: bool = False) -> str:
    skills = [
        'guardar e recuperar conhecimentos que me ensinas;',
        'resolver algumas contas e deduções simples com verificadores;',
        'treinar os pesos do meu Transformer e rever experiências;',
        'experimentar crescimento neuronal, condicionado a testes;',
    ]
    skills.append('consultar a Wikipédia com fontes, quando a Internet está ativa.' if internet_enabled
                  else 'pesquisar na Wikipédia se a Internet for ativada.')
    learned = memory.recent_teacher_learnings(limit=4)
    if learned:
        examples = '\n'.join('  • ' + x for x in learned)
        learned_text = 'Algumas coisas que me ensinaste:\n' + examples
    else:
        learned_text = 'Ainda não tenho conhecimentos teus suficientes para listar aqui.'
    lines = [
        'Neste momento, tenho estas capacidades programadas:',
        *('  • ' + skill for skill in skills),
        '',
        learned_text,
        '',
        (f'Estado: {memory.teacher_lesson_count()} exemplos ensinados por ti, '
         f'{memory.reasoning_fact_count()} relações guardadas e '
         f'{core.steps} passos de treino neuronal.'),
        'Limite: ainda não compreendo português livremente. Lembrar factos ou fazer uma dedução simbólica não prova compreensão neuronal.',
    ]
    return '\n'.join(lines)
