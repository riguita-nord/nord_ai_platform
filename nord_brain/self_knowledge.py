"""Grounded self-report from actual runtime capabilities and teacher memory.

The small Transformer cannot introspect its own weights. These descriptions
come from program capabilities and persisted records, not neural awareness.
"""
from __future__ import annotations


def report(memory, core, *, internet_enabled: bool = False) -> str:
    skills = [
        'guardar e recuperar conhecimentos que me ensinas;',
        'resolver algumas contas e deduções simples com verificadores;',
        'treinar os pesos do meu Transformer e rever experiências;',
        'simular ações alternativas numa máquina virtual e aprender a prever consequências;',
        'experimentar crescimento neuronal, condicionado a testes;',
    ]
    skills.append('consultar fontes oficiais selecionadas e treinar com excertos identificados, quando a Internet está ativa.' if internet_enabled
                  else 'pesquisar fontes oficiais selecionadas se a Internet for ativada.')
    hypothesis_stats=memory.hypothesis_stats()
    hypothesis_text=(f'Hipóteses matemáticas: {hypothesis_stats.get("pending",0)} por testar, '
                     f'{hypothesis_stats.get("trained",0)} verificadas e treinadas, '
                     f'{hypothesis_stats.get("rejected",0)} rejeitadas.')
    neural_metric = memory.get_meta('math_neural_accuracy')
    math_progress = (f'Matemática pela própria rede em testes novos: {neural_metric} acertos '
                     '(sem consultar o verificador).'
                     if neural_metric else
                     'Matemática pela própria rede: ainda sem avaliação de generalização.')
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
        math_progress,
        hypothesis_text,
        f'Excertos de fontes primárias usados no treino: {memory.source_count()} (não verificação independente).',
        learned_text,
        '',
        (f'Estado: {memory.teacher_lesson_count()} exemplos ensinados por ti, '
         f'{memory.reasoning_fact_count()} relações guardadas e '
         f'{core.steps} passos de treino neuronal.'),
        'Limite: ainda não compreendo português livremente. Lembrar factos ou fazer uma dedução simbólica não prova compreensão neuronal.',
    ]
    return '\n'.join(lines)
