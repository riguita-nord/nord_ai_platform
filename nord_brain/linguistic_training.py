"""Treino linguístico progressivo PT-PT, no Transformer Nord existente.

Combina modelação causal de texto original com respostas, continuações de
diálogo e revisão de competências anteriores. Não é outro modelo, não usa
IA externa e não treina com respostas inventadas pela própria Nord.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .conversation_corpus import TRAINING as NATURAL_TRAIN, HOLDOUT as NATURAL_HOLDOUT
from .dialogue_training import curriculum as contextual_curriculum
from .portuguese_training import curriculum as portuguese_curriculum
from .dialogue_format import training_prompt
from .contextual_dataset import examples as contextual_examples


# Exemplos humanos originais: linguagem quotidiana, composicionalidade,
# significados, descrições e turnos contextualizados. Nunca são factos pessoais.
TRAIN = (
    ('Olá, tudo bem?', 'Olá! Estou por aqui e pronto para conversar. E contigo, está tudo bem?'),
    ('Boas! Como vão as coisas?', 'Olá! Podemos conversar tranquilamente. Como correu o teu dia?'),
    ('Oi! Estás aí?', 'Estou por aqui. Queres conversar sobre alguma coisa em particular?'),
    ('Estou bem, obrigado.', 'Ainda bem! Conta-me uma coisa interessante que te aconteceu hoje.'),
    ('Hoje estou um pouco cansado.', 'Percebo. Preferes conversar com calma ou deixar os projetos para mais tarde?'),
    ('Tenho uma ideia nova.', 'Conta-me a ideia e podemos explorar os detalhes sem pressa.'),
    ('Gostava de perceber melhor.', 'Posso explicar com palavras mais simples e depois dar um exemplo.'),
    ('Não era bem isso que queria dizer.', 'Obrigado pela correção. Explica-me o que pretendias e ajusto a resposta.'),
    ('Concordas com tudo?', 'Não necessariamente. Podemos comparar ideias e reconhecer quando discordamos.'),
    ('Escreve uma frase com a palavra construir.', 'Podemos construir uma ponte, uma aplicação ou uma amizade duradoura.'),
    ('Qual é o significado de construir?', 'Construir significa criar ou montar algo, juntando partes ou desenvolvendo uma ideia.'),
    ('Escreve uma frase com descobrir.', 'Ao explorar a biblioteca, a criança conseguiu descobrir um livro antigo.'),
    ('Explica o verbo explorar.', 'Explorar é investigar um lugar, uma ideia ou uma possibilidade com curiosidade.'),
    ('A palavra banco tem sempre o mesmo sentido?', 'Não. Um banco pode ser um assento ou uma instituição financeira.'),
    ('O que muda entre correr e correu?', 'Correr está no infinitivo; correu indica uma ação concluída no passado.'),
    ('Como se forma uma pergunta clara?', 'Uma pergunta clara identifica o assunto e diz o que se pretende saber.'),
    ('Transforma esta ideia numa frase: chuva e janela.', 'A chuva batia suavemente na janela enquanto a cidade adormecia.'),
    ('Descreve uma manhã de inverno.', 'A rua estava fria e silenciosa, e o nevoeiro escondia as casas ao longe.'),
    ('Descreve um jardim ao fim da tarde.', 'A luz dourada atravessava as árvores e os pássaros regressavam aos ramos.'),
    ('Imagina um robô numa biblioteca.', 'O robô encontrou uma estante esquecida e começou a investigar os livros.'),
    ('Inventa o início de uma aventura.', 'Quando o farol se apagou, uma pequena embarcação surgiu no horizonte.'),
    ('Como criarias uma personagem curiosa?', 'Daria à personagem perguntas difíceis e vontade de explorar o desconhecido.'),
    ('Desenvolve uma ideia sem a repetir.', 'Podemos acrescentar consequências, exemplos e perspetivas diferentes à ideia inicial.'),
    ('Qual é a diferença entre facto e opinião?', 'Um facto pode ser verificado; uma opinião expressa uma avaliação ou preferência.'),
    ('Como sabes quando uma explicação é insuficiente?', 'Procuro as partes que faltam e peço esclarecimentos sobre o objetivo.'),
    ('Quero uma explicação mais acessível.', 'Podemos começar com um exemplo concreto e avançar para os detalhes.'),
    ('Fala sobre aprender com os erros.', 'Uma tentativa falhada pode mostrar o que devemos rever na próxima experiência.'),
    ('A frase parece confusa.', 'Podemos reorganizar as ideias e escolher palavras mais precisas.'),
    ('Tu: Olá!\nNord: Olá! Como estás?\nTu: Estou bem, e contigo?',
     'Obrigado por perguntares! Estou pronto para conversar. Sobre o que queres falar?'),
    ('Tu: Estou a criar uma história.\nNord: Quem é a personagem?\nTu: Uma rapariga curiosa.',
     'Podemos dar-lhe um objetivo e um mistério que a motive a explorar.'),
    ('Tu: Quero aprender Python.\nNord: Podemos começar pelo básico.\nTu: E depois?',
     'Depois podemos praticar pequenos programas e rever os erros encontrados.'),
    ('Tu: Ontem visitei uma aldeia.\nNord: O que viste?\nTu: Uma ponte antiga.',
     'Essa ponte pode ter uma história interessante. Sabes quando foi construída?'),
    ('Tu: Tenho uma aplicação.\nNord: O que faz?\nTu: Organiza tarefas.',
     'Podemos começar por ordenar as tarefas por prioridade e prazo.'),
    ('Tu: Estávamos a falar de música.\nNord: Que estilo preferes?\nTu: Afinal, quero falar de livros.',
     'Claro, mudamos para livros. Que género te desperta mais curiosidade?'),
)

# Avaliação independente: não usar estes pedidos nem estas respostas no treino.
# O teste revela perda de previsão, NÃO certifica compreensão nem criatividade.
HOLDOUT = (
    ('Olá! Como tem corrido o teu dia?', 'Olá! Podemos conversar um pouco. O que gostarias de partilhar hoje?'),
    ('Tudo bem por aí?', 'Estou disponível para conversar e ajudar. Como te sentes hoje?'),
    ('Consegues dizer isto de outra maneira?', 'Posso reformular a ideia e usar exemplos que sejam mais fáceis de compreender.'),
    ('Faz uma frase sobre um gato e uma janela.', 'O gato observou a chuva pela janela antes de adormecer no sofá.'),
    ('Explica a diferença entre observar e imaginar.', 'Observar é reparar no que acontece; imaginar é criar possibilidades na mente.'),
    ('Descreve uma praça depois da chuva.', 'As pedras brilhavam à luz dos candeeiros e o ar cheirava a terra molhada.'),
    ('Cria o início de uma história sobre um mapa.', 'O mapa apareceu dentro de um livro que ninguém abria há muitos anos.'),
    ('O que fazes quando uma palavra tem vários significados?', 'Analiso a frase e procuro o sentido que melhor corresponde ao contexto.'),
    ('Tu: Estou a desenhar uma casa.\nNord: Como será?\nTu: Pequena e luminosa.',
     'Podemos imaginar janelas amplas e uma divisão confortável junto ao jardim.'),
    ('Tu: Gosto de caminhar.\nNord: Onde costumas ir?\nTu: Hoje quero falar de fotografia.',
     'Claro, mudamos de assunto. Que tipo de fotografias gostas de tirar?'),
    ('Tu: Quero desenvolver uma ideia.\nNord: Qual é o tema?\nTu: Já não me lembro.',
     'Podemos rever o que foi dito antes ou escolher um novo ponto de partida.'),
    ('Tu: A aldeia fica junto ao rio.\nNord: Parece tranquila.\nTu: Descreve-a melhor.',
     'Podemos imaginar casas de pedra, caminhos estreitos e árvores junto à margem.'),
)


DIALOGUE_TRAIN = (
    ([('user','Olá'),('assistant','Olá! Como estás?'),('user','Tou bem e tu?')],
     'Ainda bem! Estou por aqui e pronto para conversar.'),
    ([('user','Oi'),('assistant','Olá! Como estás?'),('user','Tou bem pois')],
     'Ainda bem! Sobre o que gostavas de conversar?'),
    ([('user','Olá'),('assistant','Sobre o que te apetece falar?'),('user','Sobre tudo')],
     'Podemos falar de tudo um pouco. Preferes escolher um tema?'),
    ([('user','Quero conversar'),('assistant','Preferes tecnologia, música ou outro assunto?'),('user','Outro assunto, mas que sugeres tu?')],
     'Sugiro viagens! Se pudesses visitar qualquer país, qual escolherias?'),
    ([('user','Quero conversar'),('assistant','Preferes tecnologia ou música?'),('user','Escolhe tu')],
     'Podemos falar de viagens. Qual é o teu destino de sonho?'),
    ([('user','Quero aprender'),('assistant','Podemos falar de ciência ou história.'),('user','Prefiro história')],
     'Boa escolha! Queres explorar a história de Portugal ou de outro país?'),
)
DIALOGUE_HOLDOUT = (
    ([('user','Boas'),('assistant','Como estás hoje?'),('user','Estou bem, e contigo?')],
     'Ainda bem! Estou pronto para conversar.'),
    ([('user','Vamos conversar'),('assistant','Preferes música ou cinema?'),('user','Outro tema, sugere tu')],
     'Podemos falar de viagens. Que lugar gostavas de conhecer?'),
)

def _checks(brain):
    from .conversation_corpus import HOLDOUT as natural
    from .dialogue_training import curriculum as contextual
    from .portuguese_training import curriculum as portuguese
    context = contextual()[1][::10][:6]
    general = portuguese()[1][::3][:6]
    teacher = brain.memory.db.execute(
        "SELECT question,answer FROM lessons WHERE origin='teacher' "
        "ORDER BY id DESC LIMIT 4"
    ).fetchall()
    older = brain.memory.tests(4)
    dialogue = tuple((training_prompt(turns, brain.core.model.config.context), answer) for turns, answer in DIALOGUE_HOLDOUT)
    groups = (HOLDOUT, natural[:5], context, general, teacher, older, dialogue,
              contextual_examples(brain.core.model.config.context, holdout=True))
    return tuple(tuple((q, a) for q, a in group if
                       not brain.memory.lesson_was_trained(q))
                 for group in groups)


def _scores(brain, groups):
    return tuple(brain.core.score(group) if group else None for group in groups)


def _approved_corpus(path):
    source = (Path(__file__).parent / 'data' / 'pt_pt_seed.txt'
              if path is None else Path(path).expanduser())
    try:
        source = source.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError('Ficheiro de português não encontrado') from exc
    if not source.is_file() or source.suffix.lower() not in ('.txt', '.md'):
        raise ValueError('Usa um ficheiro .txt ou .md aprovado por ti')
    if source.stat().st_size > 100000:
        raise ValueError('Máximo de 100 KB por sessão; divide o corpus em ficheiros')
    try:
        content = source.read_text(encoding='utf-8')
    except UnicodeError as exc:
        raise ValueError('O texto deve estar em UTF-8') from exc
    if not 180 <= len(content.encode('utf-8')) <= 100000:
        raise ValueError('São necessários 180 bytes a 100 KB de texto')
    return content


def evaluate(brain):
    """Avaliação de texto novo, diálogos e competências que não podemos perder."""
    with brain.lock:
        groups = _checks(brain)
        scores = _scores(brain, groups)
        return {
            'language_loss': round(scores[0], 4) if scores[0] is not None else None,
            'natural_loss': round(scores[1], 4) if scores[1] is not None else None,
            'context_loss': round(scores[2], 4) if scores[2] is not None else None,
            'portuguese_loss': round(scores[3], 4) if scores[3] is not None else None,
            'heldout': len(groups[0]), 'steps': brain.core.steps,
            'note': 'Perdas de previsão em frases reservadas; não provam fluência.'
        }


def train(brain, updates=12, *, path=None, rate=.00006):
    """Sessão mista: texto corrido + conversas + revisão, com rollback seguro."""
    if not 1 <= updates <= 48:
        raise ValueError('Escolhe entre 1 e 48 atualizações por sessão')
    if not 0 < rate <= .0003:
        raise ValueError('Taxa de aprendizagem inválida')
    corpus = _approved_corpus(path)
    groups = None
    with brain.lock:
        if brain.memory.training_paused():
            raise ValueError('A aprendizagem está pausada')
        groups = _checks(brain)
        if not groups[0]:
            raise ValueError('Não existem exemplos linguísticos independentes para avaliar')
        before = _scores(brain, groups)
        # Anchor to the accepted checkpoint, never to rejected trial weights.
        anchor = {name: param.detach().clone()
                  for name, param in brain.core.model.named_parameters()}
        general_train, _ = portuguese_curriculum()
        contextual_train, _ = contextual_curriculum()
        dialogue_train = tuple((training_prompt(turns, brain.core.model.config.context), answer) for turns, answer in DIALOGUE_TRAIN)
        contextual_multi = contextual_examples(brain.core.model.config.context)
        pools = (TRAIN, dialogue_train, NATURAL_TRAIN, contextual_multi,
                 contextual_train, contextual_multi, general_train)
        teacher_replay = tuple(brain.memory.db.execute(
            "SELECT question,answer FROM lessons WHERE origin='teacher' "
            "ORDER BY id DESC LIMIT 64").fetchall())
        cursor = int(brain.memory.get_meta('language_curriculum_cursor', '0'))
        start_steps = brain.core.steps
        brain.core.save()
        trained = 0
        last_loss = None
        accepted = False
        regressions = []
        after = before
        attempts = 0
        try:
            ceilings = (.05, .08, .10, .10, .05, .10, .10, .10)
            labels = ('linguagem', 'conversa natural', 'contexto', 'português',
                      'lições do professor', 'testes anteriores', 'diálogo multi-turno',
                      'interpretação contextual inédita')
            for multiplier in (1.0, .25, .0625):
                attempts += 1
                brain.core.load()
                trained = 0
                for n in range(updates):
                    index = cursor + n
                    step_rate = rate * multiplier
                    if index % 12 == 0:
                        last_loss = brain.core.train_text(corpus, updates=1,
                                                          rate=step_rate * .65)
                        trained += 1
                        continue
                    pool = (teacher_replay if teacher_replay and index % 4 == 1
                            else pools[index % len(pools)])
                    row = pool[((index * 19) // 7 + index // 3) % len(pool)]
                    question, answer = row
                    if (brain.memory.is_training_blocked(question)
                            or brain.memory.is_holdout(question)
                            or brain.memory.is_cognitive_probe(question)):
                        continue
                    last_loss = brain.core.train(question, answer, epochs=1,
                                                 rate=step_rate)
                    trained += 1
                    # Conservative weight anchoring prevents rapid drift.
                    # This is a lightweight L2-style approximation, not EWC.
                    with torch.no_grad():
                        for name, param in brain.core.model.named_parameters():
                            param.lerp_(anchor[name], .025)
                after = _scores(brain, groups)
                regressions = [
                    f'{label}: {old:.4f} -> {new:.4f} (limite +{margin:.2f})'
                    for label, old, new, margin in zip(labels, before, after, ceilings)
                    if new is not None and (not math.isfinite(new) or
                        (old is not None and new > old + margin))
                ]
                accepted = trained > 0 and not regressions
                if accepted:
                    break
            if accepted:
                brain.core.save()
                brain.memory.set_meta('language_curriculum_cursor', str(cursor + updates))
                brain.memory.metric('language_v3_heldout_loss', after[0])
                brain.memory.event('language_v3_training',
                    f'update={trained}; attempts={attempts}; heldout={before[0]:.4f}->{after[0]:.4f}', after[0])
            else:
                brain.core.load()
                brain.memory.event('language_v3_rejected',
                    'Regressão após tentativas com taxas de aprendizagem reduzidas')
            return {
                'accepted': accepted,
                'regressions': regressions,
                'attempts': attempts,
                'examples': trained,
                'steps': brain.core.steps - start_steps if accepted else 0,
                'before': round(before[0], 4),
                'after': round(after[0], 4),
                'natural_before': round(before[1], 4) if before[1] is not None else None,
                'natural_after': round(after[1], 4) if after[1] is not None else None,
                'cursor': cursor + updates if accepted else cursor,
                'last_training_loss': round(last_loss, 4) if last_loss is not None else None,
            }
        except Exception:
            brain.core.load()
            raise


def describe(result):
    status = ('Treino linguístico aceite e guardado.'
              if result['accepted'] else
              'Treino rejeitado; checkpoint anterior recuperado.')
    return (f"{status} Exemplos usados: {result['examples']}; "
            f"passos neuronais: {result['steps']}. "
            f"Tentativas adaptativas: {result.get('attempts', 1)}. "
            f"Perda linguística reservada: {result['before']} → {result['after']}. " +
            ('Avaliações que impediram o treino: ' + '; '.join(result['regressions']) + '. '
             if result.get('regressions') else '') +
            'A fluência precisa de ser testada em conversas inéditas.')
