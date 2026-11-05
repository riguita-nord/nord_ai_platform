"""Supervised multi-turn European Portuguese practice on Nord's OWN Transformer.

Unlike isolated Q&A examples, inputs have two previous turns and a new
request. Text and labels are curated synthetic pedagogical examples, never
web pages, generated model guesses, or implicit teacher corrections.
"""
from __future__ import annotations

import math

# Each entry is an invented conversational subject, a prior assistant answer
# and a specific item that makes the next turn actually refer to the context.
_CONTEXTS=(
    ('um robô que explora a cidade','Podemos usar sensores e um mapa simples.','sensores e mapas'),
    ('uma história sobre um farol','A protagonista encontrou um diário misterioso.','o diário e a protagonista'),
    ('um jogo de aventuras','O jogador pode escolher caminhos diferentes.','escolhas e caminhos'),
    ('uma aplicação de organização','A agenda pode mostrar tarefas por prioridade.','tarefas e prioridades'),
    ('uma máquina de reciclagem','A máquina deve separar os materiais com cuidado.','materiais e separação'),
    ('um jardim comunitário','Podemos distribuir espaços de cultivo pelos moradores.','plantas e moradores'),
    ('uma viagem pelo interior','O percurso pode passar por aldeias e miradouros.','paragens e percursos'),
    ('uma escola de robótica','Os alunos podem construir projetos simples.','alunos e projetos'),
    ('um pequeno motor elétrico','Vamos medir eficiência, calor e consumo.','consumo e eficiência'),
    ('uma oficina de bicicletas','Precisamos de ferramentas e manutenção regular.','ferramentas e manutenção'),
)

_ACTIONS=(
    # Phrasings for training; one never-seen phrase for independent evaluation.
    (
      ('podes desenvolver melhor', 'queria mais pormenores', 'expande a tua resposta'),
      'consegues aprofundar essa ideia',
      'Podemos desenvolver {topic} com mais detalhes sobre {detail}, sem perder o objetivo inicial.'
    ),
    (
      ('quais são os riscos', 'que problemas podem aparecer', 'e as desvantagens'),
      'o que pode correr mal nessa proposta',
      'Em {topic}, devemos verificar os riscos de {detail} antes de avançar com a solução.'
    ),
    (
      ('e se a primeira ideia falhar', 'imagina que o plano não funciona',
       'podemos testar outra hipótese'),
      'qual seria a alternativa se isso falhasse',
      'Se {topic} não funcionar como previsto, podemos experimentar outra abordagem e comparar resultados.'
    ),
    (
      ('concordas com essa escolha', 'o que achas do que decidimos',
       'qual é a tua opinião'),
      'achas mesmo que vale a pena',
      'Acho {topic} uma ideia promissora, mas avaliaria {detail} antes de tomar uma decisão.'
    ),
    (
      ('faz um resumo do que dissemos', 'consegues resumir isso',
       'diz só o essencial'),
      'explica tudo de forma mais curta',
      'Resumindo, estávamos a falar de {topic}; o próximo passo é verificar {detail}.'
    ),
    (
      ('não percebi a explicação', 'como chegaste a essa ideia',
       'podes explicar de outra maneira'),
      'porque é que escolheste esse caminho',
      'O ponto central de {topic} é experimentar {detail} e observar o que realmente acontece.'
    ),
)


def curriculum() -> tuple[tuple[tuple[str,str],...],tuple[tuple[str,str],...]]:
    """Disjoint training/holdout full conversation prompts.

    Cross-topic variations are still partially templated, so evaluation is an
    early signal, not proof of rich Portuguese understanding.
    """
    training=[]
    holdout=[]
    for topic,previous,detail in _CONTEXTS:
        history=f'Tu: Estou a pensar em {topic}.\nNord: {previous}\nTu: '
        for variants,unseen,answer in _ACTIONS:
            for query in variants:
                training.append((history+query,answer.format(topic=topic,detail=detail)))
            holdout.append((history+unseen,answer.format(topic=topic,detail=detail)))
    assert len(training)==180 and len(holdout)==60
    return tuple(training),tuple(holdout)


def _probes(brain):
    # Evaluate all reserved contextual phrases, Portuguese checks, teacher
    # corrections, and natural conversation without training on any of them.
    _,heldout=curriculum()
    valid=[row for row in heldout if not brain.memory.lesson_was_trained(row[0])]
    # Use all 60 contextual held-out variants instead of a biased subsample.
    dialogue=valid
    from .portuguese_training import curriculum as base
    old=[row for row in base()[1] if not brain.memory.lesson_was_trained(row[0])]
    teacher=brain.memory.db.execute(
        "SELECT question,answer FROM lessons WHERE origin='teacher' ORDER BY id DESC LIMIT 4"
    ).fetchall()
    other=brain.memory.tests(4)
    from .conversation_corpus import HOLDOUT
    natural=[row for row in HOLDOUT if not brain.memory.lesson_was_trained(row[0])]
    return dialogue,old,teacher,other,natural


def evaluate(brain) -> dict:
    with brain.lock:
        dialogue,old,_,_,natural=_probes(brain)
        return {'dialogue_loss':round(brain.core.score(dialogue),4) if dialogue else None,
                'portuguese_loss':round(brain.core.score(old),4) if old else None,
                'heldout_dialogues':len(dialogue),'legacy_holdouts':len(old),
                'natural_loss':round(brain.core.score(natural),4) if natural else None,
                'natural_holdouts':len(natural),
                'steps':brain.core.steps,
                'note':'Perda em exemplos nunca treinados; não prova fluência.'}


def train(brain,updates: int=12,rate: float=.00006) -> dict:
    if not 1<=updates<=48:
        raise ValueError('Usa entre 1 e 48 atualizações por sessão')
    if not 0<rate<=.0003:
        raise ValueError('Taxa de aprendizagem deve ser conservadora (máximo 0.0003)')
    rows,_=curriculum()
    from .portuguese_training import curriculum as portuguese_curriculum
    from .conversation_corpus import TRAINING
    portuguese_rows,_=portuguese_curriculum()
    with brain.lock:
        if brain.memory.training_paused():
            raise ValueError('A aprendizagem está pausada')
        dialogue,old,teacher,other,natural=_probes(brain)
        if not dialogue:
            raise ValueError('Não há frases de avaliação reservadas')
        before=tuple(brain.core.score(block) if block else None
                     for block in (dialogue,old,teacher,other,natural))
        cursor=int(brain.memory.get_meta('dialogue_training_cursor','0'))
        baseline_steps=brain.core.steps
        brain.core.save()
        trained=0
        try:
            for n in range(updates):
                # Mixed curriculum: contextual turns, natural conversation and
                # general Portuguese all update the same neural weights.
                # Each category has its own deterministic rotation.
                phase=(cursor+n)%5
                if phase==3:
                    pool=TRAINING
                elif phase==4:
                    pool=portuguese_rows
                else:
                    pool=rows
                q,a=pool[((cursor+n)//5 + (cursor+n)%3 * 17)%len(pool)]
                if (brain.memory.is_holdout(q) or brain.memory.is_training_blocked(q)
                        or brain.memory.is_cognitive_probe(q)):
                    continue
                brain.core.train(q,a,epochs=1,rate=rate)
                trained+=1
            after=tuple(brain.core.score(block) if block else None
                        for block in (dialogue,old,teacher,other,natural))
            accepted=trained>0 and all(
                math.isfinite(current) and (
                    current<=baseline*multiplier+absolute)
                for baseline,current,multiplier,absolute in zip(
                    before,after,(1.025,1.035,1.015,1.045,1.035),(.025,.035,.03,.04,.035))
                if baseline is not None and current is not None)
            if accepted:
                brain.core.save()
                brain.memory.set_meta('dialogue_training_cursor',str(cursor+updates))
                brain.memory.metric('dialogue_validation_loss',after[0])
                brain.memory.event('dialogue_training',
                    f'{trained} steps, heldout {before[0]:.3f}->{after[0]:.3f}',after[0])
            else:
                brain.core.load()
                brain.memory.event('dialogue_training_rejected',
                    f'validation {before[0]:.3f}->{after[0]:.3f}',after[0])
            return {
                'accepted':accepted,'updates':brain.core.steps-baseline_steps if accepted else 0,
                'dialogue_before':round(before[0],4),
                'dialogue_after':round(after[0],4),
                'portuguese_before':round(before[1],4) if before[1] is not None else None,
                'portuguese_after':round(after[1],4) if after[1] is not None else None,
                'natural_before':round(before[4],4) if before[4] is not None else None,
                'natural_after':round(after[4],4) if after[4] is not None else None,
                'cursor':cursor+updates if accepted else cursor,
            }
        except Exception:
            brain.core.load()
            raise


def describe(result: dict) -> str:
    return ('Treino aceite; pesos guardados.' if result['accepted'] else
            'Treino rejeitado; pesos anteriores restaurados.')+(
            f" Passos reais guardados: {result['updates']}. "
            f"Perda de diálogo: {result['dialogue_before']} → {result['dialogue_after']}; "
            f"perda de português anterior: {result['portuguese_before']} → "
            f"{result['portuguese_after']}. "
            'São métricas de previsão, não uma medição de conversa fluente.')
