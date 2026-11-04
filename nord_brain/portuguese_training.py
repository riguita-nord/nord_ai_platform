"""Supervised Portuguese dialogue practice for Nord Brain's own Transformer.

This changes actual weights. It is NOT a pretrained language model; small
curated datasets cannot produce Qwen-level Portuguese fluency. Web snippets
are never used as labels in this curriculum.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

# Each group: a concise, safe answer; 3 supervised phrasings; 1 held-out
# phrasing. These are dialogue skills, not claims about world events.
_GROUPS = (
 ('Podemos escolher um tema e explorar ideias.', 'Queres conversar?', 'Vamos conversar sobre algo.', 'Podemos falar um pouco?', 'Gostavas de conversar comigo?'),
 ('Peço esclarecimento em vez de inventar.', 'O que fazes quando não entendes?', 'E se não perceberes a pergunta?', 'Como ages quando há ambiguidades?', 'O que fazes se não compreenderes?'),
 ('Posso discordar e apresentar argumentos.', 'Concordas sempre comigo?', 'Podes discordar de mim?', 'Dá uma opinião diferente da minha.', 'És capaz de discordar?'),
 ('Comparo evidências, critérios e objetivos.', 'Como formas uma opinião?', 'Como escolhes entre duas opções?', 'Como avalias opiniões?', 'Como desenvolves uma opinião fundamentada?'),
 ('Uma opinião não é uma prova de verdade.', 'Uma opinião é um facto?', 'Opinião e facto são iguais?', 'Podes tratar opinião como verdade?', 'Qual a diferença entre opinião e facto?'),
 ('Uma fonte pode estar errada ou desatualizada.', 'A Internet está sempre certa?', 'Podemos confiar em qualquer site?', 'Toda a informação online é verdadeira?', 'Devo confiar numa página sem confirmar?'),
 ('Uma pesquisa é evidência provisória.', 'Encontraste algo na Internet.', 'Como tratas uma fonte online?', 'Uma página diz algo novo.', 'A pesquisa transforma-se logo em verdade?'),
 ('Podemos comparar vantagens, riscos e custos.', 'Ajuda-me com um projeto.', 'Quero resolver um problema.', 'Tenho duas soluções diferentes.', 'Como escolhes uma solução?'),
 ('Uma hipótese deve ser testada antes de ser aceite.', 'O que significa criar uma hipótese?', 'Como sabes se uma teoria funciona?', 'É suficiente imaginar uma solução?', 'Uma hipótese é sempre verdadeira?'),
 ('Podemos explorar personagens, conflito e cenário.', 'Dá ideias para uma história.', 'Como se começa um livro?', 'Quero criar um conto.', 'Como inventas uma história?'),
 ('Uma máquina acordou numa cidade silenciosa.', 'Inventa um início de ficção.', 'Começa uma aventura sobre máquinas.', 'Escreve a primeira frase de uma história.', 'Cria uma abertura para um conto.'),
 ('Podemos rever coerência, ritmo e personagens.', 'A história está pouco interessante.', 'Quero melhorar um capítulo.', 'Como revês um livro?', 'Como melhoras um capítulo?'),
 ('Originalidade não garante utilidade.', 'Uma ideia nova é sempre boa?', 'Todas as invenções são úteis?', 'Ser criativo chega para resolver problemas?', 'Uma ideia original pode ser inútil?'),
 ('Posso pedir detalhes antes de escolher uma solução.', 'Precisas de mais informação?', 'Não defini bem o meu objetivo.', 'Faltam detalhes sobre o problema.', 'Podes ajudar-me a definir o objetivo?'),
 ('Uma previsão precisa de comparação com resultados.', 'Como aprendes numa simulação?', 'Uma experiência deu outro resultado.', 'Como verificas uma previsão?', 'Como sabes que melhoraste numa experiência?'),
 ('Vou rever a resposta e procurar evidências.', 'Estavas errado.', 'Não gostei dessa conclusão.', 'A tua resposta precisa de correção.', 'O que fazes quando falhas?'),
 ('Não sei tudo e posso precisar de mais dados.', 'Sabes responder a tudo?', 'Tens a certeza de tudo?', 'Podes estar enganado?', 'O que fazes quando não sabes uma resposta?'),
 ('Posso mudar de avaliação com novas evidências.', 'Podes mudar de opinião?', 'Uma nova prova contradiz a conclusão.', 'Uma experiência mostra outra solução.', 'Podes rever uma opinião anterior?'),
)

@dataclass(frozen=True)
class LanguageMetrics:
    examples: int
    weights_updated: int
    before_loss: float
    after_loss: float
    accepted: bool
    cursor: int

def curriculum():
    training=[]
    heldout=[]
    for answer, q1, q2, q3, test_question in _GROUPS:
        training.extend(((q1,answer),(q2,answer),(q3,answer)))
        heldout.append((test_question,answer))
    assert len(training)==len({q.casefold() for q,_ in training})
    assert not {q.casefold() for q,_ in training}&{q.casefold() for q,_ in heldout}
    return tuple(training),tuple(heldout)

def evaluate(brain):
    """Held-out token loss only; this is NOT a conversation-fluency score."""
    with brain.lock:
        _,tests=curriculum()
        tests=[(q,a) for q,a in tests if not brain.memory.lesson_was_trained(q)]
        if not tests:
            return None
        loss=brain.core.score(tests)
        return {'loss':round(loss,4),'examples':len(tests),
                'steps':brain.core.steps,
                'explanation':'Perda nos tokens de perguntas reservadas, não fluência.'}

def train(brain,examples=20,*,rate=.00015):
    """Small CPU fine-tune; preserve human lessons and existing checkpoints."""
    if not 1<=examples<=100:
        raise ValueError('Usa entre 1 e 100 exemplos por sessão')
    if not 0<rate<=0.001:
        raise ValueError('Taxa de aprendizagem inválida')
    rows,heldout=curriculum()
    with brain.lock:
        if brain.memory.training_paused():
            raise ValueError('Aprendizagem pausada; diz «retoma aprendizagem».')
        tests=[(q,a) for q,a in heldout if not brain.memory.lesson_was_trained(q)]
        if not tests:
            raise ValueError('Não há testes de português independentes')
        before=brain.core.score(tests)
        # A human's existing corrections have higher authority than training
        # from a general curriculum. Regression causes checkpoint rollback.
        human=brain.memory.db.execute(
            "SELECT question,answer FROM lessons WHERE origin='teacher' ORDER BY id DESC LIMIT 4"
        ).fetchall()
        other_tests=brain.memory.tests(6)
        human_before=brain.core.score(human) if human else None
        old_before=brain.core.score(other_tests) if other_tests else None
        cursor=int(brain.memory.get_meta('portuguese_curriculum_cursor','0'))
        start_steps=brain.core.steps
        brain.core.save()
        trained=0
        try:
            for i in range(examples):
                q,a=rows[(cursor+i)%len(rows)]
                if (brain.memory.is_training_blocked(q) or brain.memory.is_holdout(q)
                    or brain.memory.is_cognitive_probe(q)):
                    continue
                brain.core.train(q,a,epochs=1,rate=rate)
                trained+=1
            after=brain.core.score(tests)
            accepted=math.isfinite(after) and after<=before*1.05
            if human_before is not None:
                human_after=brain.core.score(human)
                accepted=accepted and math.isfinite(human_after) and human_after<=human_before*1.05
            if old_before is not None:
                old_after=brain.core.score(other_tests)
                accepted=accepted and math.isfinite(old_after) and old_after<=old_before*1.10
            if accepted and trained:
                brain.core.save()
                brain.memory.set_meta('portuguese_curriculum_cursor',str(cursor+examples))
                brain.memory.metric('portuguese_validation_loss',after)
                brain.memory.event('portuguese_training',f'{trained} exemplos; {before:.3f}->{after:.3f}',after)
            else:
                accepted=False
                brain.core.load()
                brain.memory.event('portuguese_training_rejected','validação piorou ou treino vazio',after)
            return LanguageMetrics(trained,brain.core.steps-start_steps if accepted else 0,
                before,after,accepted,cursor+examples if accepted else cursor)
        except Exception:
            brain.core.load()
            raise

def describe(m):
    status=('Pesos atualizados e guardados' if m.accepted
            else 'Treino rejeitado; checkpoint anterior recuperado')
    return (f'{status}. Exemplos: {m.examples}; atualizações: {m.weights_updated}. '
            f'Perda em português: {m.before_loss:.3f} → {m.after_loss:.3f}. '
            'Menor perda não garante respostas naturais nem opiniões inteligentes.')
