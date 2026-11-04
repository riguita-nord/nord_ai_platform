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



def respond(brain, kind: str) -> str:
    """Report actual program state, not imagined access to neural thoughts.

    This is introspection of the local *software and persisted records*. It
    is not neural self-awareness, subjective experience, or access to raw
    unobservable thought processes. Do not treat configuration as a guarantee
    of real-time external connectivity.
    """
    import os
    from . import __version__

    memory, core = brain.memory, brain.core
    config = core.model.config
    internet_enabled = os.getenv('NORD_INTERNET', 'on').lower() not in (
        'off', '0', 'false', 'no')
    autonomy = brain.autonomy.settings
    if kind == 'summary':
        overview = (f'Sou a Nord Brain v{__version__}, um Transformer experimental com '
                    f'{core.parameter_count} parâmetros e {config.layers} camadas. '
                    'As minhas respostas combinam treino neuronal, memória local e '
                    'verificadores especializados; isto não é autoconsciência.\n')
        return overview + report(memory, core, internet_enabled=internet_enabled)
    if kind == 'identity':
        return (f'Sou a Nord Brain, versão {__version__}: uma IA experimental '
                'baseada num Transformer próprio, sem pesos pré-treinados. '
                'Respondo no terminal e aprendo com exemplos. Não sou uma pessoa.')
    if kind == 'creator':
        return ('Sou um projeto da Nord Lab. O meu código é desenvolvido e '
                'mantido no repositório riguita-nord/nord_ai_platform. '
                'Não consigo verificar automaticamente todos os autores individuais.')
    if kind == 'version':
        return (f'A minha versão de software é {__version__}. Não consigo '
                'confirmar se há uma versão mais recente sem consultar o repositório.')
    if kind == 'age':
        return ('Não tenho uma data de nascimento biológica nem uma data de criação '
                'verificada na minha memória. Sei indicar a versão do software, '
                'mas não posso inventar a minha idade.')
    if kind == 'location':
        return ('Estou a executar neste processo Python, na máquina que iniciou '
                'a Nord Brain. Não conheço a localização física da máquina.')
    if kind == 'purpose':
        return ('O meu objetivo enquanto software é conversar contigo, aprender '
                'com exemplos, guardar conhecimento, experimentar previsões '
                'e melhorar nos testes, sempre dentro das limitações do código.')
    if kind == 'limitations':
        return ('Ainda não compreendo português livremente como um modelo grande '
                'treinado. Posso confundir perguntas e instruções; os meus '
                'verificadores só resolvem tipos limitados de contas e relações. '
                'Memorizar uma frase ou obter um resultado simbólico não prova '
                'que o meu Transformer generalizou. Não conheço tudo, não '
                'controlo a máquina sem ferramentas e não devo inventar factos.')
    if kind == 'architecture':
        return (f'O meu cérebro de texto é um Transformer decoder-only com '
                f'{config.layers} camadas, largura {config.d_model}, '
                f'{config.heads} cabeças de atenção e janela de contexto '
                f'{config.context} tokens UTF-8. Usa RoPE, GQA, RMSNorm e SwiGLU. '
                'Os meus pesos são treinados localmente. Para algumas respostas '
                'uso memória e verificadores separados, e não raciocínio neuronal.')
    if kind == 'size':
        return (f'Tenho {core.parameter_count} parâmetros treináveis no Transformer '
                f'de texto, {config.layers} camadas, largura {config.d_model}, '
                f'MLP com {core.hidden_size} canais e contexto {config.context}. '
                'Parâmetros e canais não são o mesmo que neurónios biológicos.')
    if kind == 'memory':
        concepts = memory.teacher_concepts(limit=5)
        summary = (f'Guardo dados localmente no ficheiro {memory.path}. '
                   f'Tenho {memory.teacher_lesson_count()} exemplos de ensino humano '
                   f'e {memory.reasoning_fact_count()} relações ensinadas. '
                   'Não retenho necessariamente todas as mensagens da conversa; '
                   'guardo apenas os registos que o programa persiste.')
        if concepts:
            summary += '\nAlguns conceitos ensinados: ' + '; '.join(
                f'{term}: {description[:110]}' for term, description in concepts)
        return summary
    if kind == 'learned':
        examples = memory.recent_teacher_learnings(limit=8)
        if not examples:
            return ('Ainda não tenho conceitos ensinados suficientes para listar. '
                    f'Tenho {memory.teacher_lesson_count()} exemplos de ensino guardados.')
        return ('Estes são exemplos de conhecimentos que me ensinaste, guardados '
                'na memória (não necessariamente compreendidos pela rede):\n' +
                '\n'.join('  • ' + item for item in examples))
    if kind == 'training':
        paused = memory.training_paused()
        enabled = getattr(autonomy, 'training', False)
        state = ('em pausa por tua decisão' if paused else
                 'configurado para treino autónomo' if enabled else
                 'com o treino autónomo desativado')
        return (f'Já fiz {core.steps} passos de treino no Transformer de texto '
                f'e tenho {memory.teacher_lesson_count()} exemplos de ensino humano. '
                f'Neste momento o treino está {state}. '
                'Isso não significa que esteja a atualizar os pesos neste exato segundo; '
                'só conto aprendizagem validada em testes independentes.')
    if kind == 'internet':
        return (('A pesquisa pública está autorizada na configuração. '
                 'Posso tentar consultar fontes e indicar a origem, mas não posso '
                 'garantir conectividade ou veracidade sem verificar cada fonte.')
                if internet_enabled else
                ('A pesquisa está desativada por NORD_INTERNET=off. '
                 'Posso continuar a usar conhecimento e exemplos guardados localmente.'))
    if kind == 'experiments':
        world = brain.world
        hyp = memory.hypothesis_stats()
        return (f'Tenho {hyp.get("pending",0)} hipóteses matemáticas pendentes, '
                f'{hyp.get("trained",0)} treinadas e {hyp.get("rejected",0)} rejeitadas. '
                f'O meu modelo de mundo tem {world.observations} observações, '
                f'{world.episodes} cenários e {world.train_steps} passos de treino. '
                'São simulações e verificadores computacionais, não experiência real do mundo.')
    if kind == 'experience':
        return ('Sou um programa de IA, não um ser humano. Não tenho evidência de '
                'experiência subjetiva ou sentimentos; as minhas respostas sobre '
                'emoções são texto produzido pelo software, não relatos pessoais.')
    return 'Não tenho informação verificada para responder a essa pergunta sobre mim.'
