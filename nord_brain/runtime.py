"""The Nord Brain terminal runtime: conversational teaching and autonomous training.

Responses distinguish symbolic knowledge from unverified neural generation.
The internet is treated as untrusted research material, never a training label.
"""
from __future__ import annotations
from dataclasses import asdict
import os
from pathlib import Path
import threading

from .core import NeuralCore, TransformerConfig
from .storage import Memory
from .vocabulary import initialize
from .conversation import Intent, interpret, is_self_knowledge_request
from .autonomy import Autonomy
from .internet import search
from .reasoning import ReasoningEngine, parse_fact
from .self_knowledge import report as self_knowledge_report


class Brain:
    def __init__(self,data_dir: Path, *, start_autonomy: bool = True):
        self.data_dir=Path(data_dir)
        self.data_dir.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()
        self.memory=Memory(self.data_dir/'memory.sqlite3')
        self.vocabulary=initialize(self.data_dir/'vocabulary.json')
        profiles={
            'tiny':TransformerConfig(d_model=64,layers=2,heads=2,kv_heads=1,intermediate_size=192,context=128),
            'small':TransformerConfig(),
            'medium':TransformerConfig(d_model=192,layers=6,heads=6,kv_heads=3,intermediate_size=512,context=384),
        }
        profile=os.environ.get('NORD_BRAIN_PRESET','small').lower()
        if profile not in profiles:
            raise ValueError('NORD_BRAIN_PRESET deve ser tiny, small ou medium')
        self.core=NeuralCore(self.data_dir/'brain-v2.pt',config=profiles[profile])
        self.reasoner=ReasoningEngine(self.memory)
        self.last_question=None
        self.awaiting_explanation=False
        self.autonomy=Autonomy(self)
        if start_autonomy:self.autonomy.start()

    def _learn(self,question,answer,epochs=1,origin='teacher'):
        """Actual online gradient update followed by durable persistence."""
        question,answer=question.strip(),answer.strip()
        if not 1<=len(question)<=250 or not 1<=len(answer)<=500:
            raise ValueError('Exemplo vazio ou demasiado longo (250/500 caracteres)')
        if self.memory.is_holdout(question):
            raise ValueError('Este exemplo está reservado para avaliação independente')
        self.core.save()
        try:
            loss=self.core.train(question,answer,epochs=epochs)
            self.core.save()
        except Exception:
            self.core.load()
            raise
        # Commit the example only after neural training succeeded.
        self.memory.teach(question,answer,origin=origin)
        self.memory.metric('train_loss',loss)
        return loss

    def teach(self,question,answer,epochs=2):
        with self.lock:return self._learn(question,answer,epochs=epochs)

    def answer(self,question,neural_only=False):
        with self.lock:
            question=question.strip()
            self.last_question=question
            if not neural_only and (is_self_knowledge_request(question) or
                    self.memory.resolve_intent_alias(question)=='self_knowledge'):
                return self_knowledge_report(
                    self.memory, self.core,
                    internet_enabled=os.getenv('NORD_INTERNET','off').lower() in ('on','1','true')
                ), 'relatório local de capacidades e memória'
            if not neural_only:
                known=self.memory.exact(question)
                if known is not None:
                    return known,'memória ensinada (não é generalização neuronal)'
                concept=self.memory.lookup_concept(question)
                if concept:
                    return concept,'conceito na memória'
            if self.memory.lesson_count()<15 and not neural_only:
                return 'Ainda não sei responder. Podes explicar-me?','sem treino suficiente'
            return (self.core.generate(question) or 'Ainda não sei responder.'),'rede neuronal experimental, não verificada'

    def chat(self,message: str) -> tuple[str,str]:
        """Teacher-facing conversation. Intents are explicit, limited heuristics."""
        with self.lock:
            intent=interpret(message,self.last_question,self.awaiting_explanation)
            # Persistent, teacher-defined phrasing for a supported dynamic intent.
            # Never reuse an old memorised answer for a self-report.
            if (intent.kind in ('question','definition_question','explanation')
                    and self.memory.resolve_intent_alias(message)=='self_knowledge'):
                intent=Intent('self_knowledge')
            if intent.kind=='intent_instruction':
                self.awaiting_explanation=False
                self.last_question=None
                if intent.topic!='self_knowledge':
                    return ('Percebi que queres definir como devo interpretar uma frase, '
                            'mas ainda só consigo guardar regras sobre as minhas capacidades '
                            'e o que aprendi. Podes explicar-me isso dessa forma?'), 'instrução não suportada'
                self.memory.set_intent_alias(intent.question, 'self_knowledge')
                return (f'Percebi. Quando disseres «{intent.question}», vou mostrar as '
                        'minhas capacidades reais e o que já me ensinaste, em vez de '
                        'repetir esta explicação.'), 'interpretação guardada, não treino neuronal'
            if intent.kind=='self_knowledge':
                self.awaiting_explanation=False
                self.last_question=None
                return (self_knowledge_report(
                    self.memory, self.core,
                    internet_enabled=os.getenv('NORD_INTERNET','off').lower() in ('on','1','true')
                ), 'relatório local de capacidades e memória')
            if intent.kind=='empty':return 'Diz-me alguma coisa.','terminal'
            if intent.kind=='greeting':return 'Olá! Estou a aprender. Podes ensinar-me algo ou fazer uma pergunta.','interface inicial'
            if intent.kind=='farewell':return 'Até à próxima! Vou guardar aquilo que aprendi.','interface inicial'
            if intent.kind=='progress':
                s=self.status()
                return (f"Já fiz {s['passos_de_treino']} passos de treino, tenho {s['exemplos_ensinados']} exemplos e "
                        f"{s['parametros']} parâmetros. As respostas da memória não provam raciocínio neuronal."),'métricas reais'
            if intent.kind in ('teach','identity','fact','correction','explanation'):
                self.awaiting_explanation=False
                if intent.kind=='identity':
                    self.memory.concept('nome do professor',intent.answer)
                if intent.kind=='fact':
                    self.memory.concept(intent.topic,intent.answer,origin='teacher')
                self._learn(intent.question,intent.answer,epochs=1,origin='teacher')
                if intent.kind=='fact':
                    self.reasoner.learn(message)
                if intent.kind=='correction':
                    return 'Obrigado pela correção. Atualizei os pesos e vou rever este exemplo.','aprendizagem neuronal + memória'
                if intent.kind=='identity':
                    return f'Prazer! Aprendi que te chamas {intent.answer}.','aprendizagem neuronal + memória'
                if intent.kind=='fact':
                    return f'Entendido. Aprendi a descrição de {intent.topic}.','aprendizagem neuronal + memória'
                return 'Entendido. Guardei o exemplo e atualizei os pesos neuronais.','aprendizagem neuronal + memória'
            # Unambiguous declarative teacher facts can teach relations as well.
            # The symbolic graph is distinct from the Transformer weights.
            if intent.kind=='question' and not message.strip().endswith('?'):
                fact=parse_fact(message)
                if fact:
                    self._learn(f'O que sabes sobre {fact.subject}?',fact.statement,epochs=1,origin='teacher')
                    self.reasoner.learn(message)
                    self.awaiting_explanation=False
                    return 'Aprendi essa relação. Vou tentar aplicá-la a outras perguntas.', 'premissa ensinada + treino neuronal'
            # Verified composition before a neural guess: no pretending that
            # deterministic math or graph traversal is neural generalisation.
            if intent.kind in ('question','definition_question'):
                # Existing explicitly taught answers retain priority over inference.
                direct=self.memory.exact(intent.question or message) if intent.kind=='question' else None
                if direct is not None:
                    self.last_question=message.strip()
                    self.awaiting_explanation=False
                    return direct,'memória ensinada (não é generalização neuronal)'
                # Keep the teacher's direct definition preferred over a graph listing.
                proof=(None if intent.kind=='definition_question' and self.memory.lookup_concept(intent.topic)
                       else self.reasoner.solve(message))
                if proof:
                    self.last_question=message.strip()
                    self.awaiting_explanation=proof.method=='incomplete'
                    if proof.method=='incomplete':self.memory.unknown(message[:90])
                    else:self.memory.event('reasoning',proof.method,float(len(proof.steps)))
                    return proof.answer, 'raciocínio verificável (simbólico, não neuronal)' if proof.method!='incomplete' else 'informação insuficiente'
            if intent.kind=='search':
                if os.getenv('NORD_INTERNET','off').lower() not in ('on','1','true'):
                    return 'Ainda não tenho acesso à Internet. Preciso que seja ativada na configuração.','ferramenta indisponível'
                results=search(intent.topic)
                for item in results:self.memory.observation(intent.topic,item['title'],item['url'],item['summary'])
                if not results:
                    return 'Pesquisei mas não encontrei resultados úteis.','pesquisa sem resultados'
                item=results[0]
                return (f"Encontrei isto sobre {intent.topic}: {item['summary'][:350]}\nFonte: {item['url']}\n"
                        'Ainda não confirmei esta informação, por isso não a usei para treinar os pesos.'),'Internet (fonte não verificada)'
            if intent.kind=='definition_question':
                self.last_question=f'O que é {intent.topic}?'
                known=self.memory.lookup_concept(intent.topic)
                if known:
                    self.awaiting_explanation=False
                    return known,'conceito ensinado'
                known=self.memory.exact(self.last_question)
                if known:
                    self.awaiting_explanation=False
                    return known,'memória ensinada'
                ref=self.memory.observation_for_topic(intent.topic)
                if ref:
                    return f'Encontrei uma fonte, ainda não confirmada: {ref[2][:250]}\nFonte: {ref[1]}','pesquisa não verificada'
                self.memory.unknown(intent.topic)
                self.awaiting_explanation=True
                return f'Ainda não sei o que é {intent.topic}. Podes explicar-me?', 'pergunta de aprendizagem'
            # Normal question, no teacher answer assumed.
            self.awaiting_explanation=False
            reply,source=self.answer(intent.question)
            if source=='sem treino suficiente':
                self.awaiting_explanation=True
                self.memory.unknown(intent.question[:90])
            return reply,source

    def practice(self,limit=20):
        with self.lock:
            examples=self.memory.lessons(limit)
            if not examples:return 0,None
            losses=[self.core.train(q,a,epochs=1) for q,a in reversed(examples)]
            self.core.save()
            value=sum(losses)/len(losses)
            self.memory.metric('practice_loss',value)
            return len(losses),value

    def evaluate(self):
        with self.lock:
            tests=self.memory.tests(10)
            if not tests:return None
            correct=0
            for question,answer in tests:
                prediction=self.core.generate(question,temperature=0).strip().casefold()
                correct+=prediction==answer.strip().casefold()
            score=correct/len(tests)
            self.memory.metric('heldout_accuracy',score)
            return correct,len(tests),score

    def grow(self,extra=16):
        with self.lock:
            self.core.grow(extra)
            self.core.save()
            return self.core.hidden_size

    def status(self):
        with self.lock:
            return {'largura_mlp':self.core.hidden_size,
                    'arquitetura':'decoder-only Transformer (RoPE, GQA, RMSNorm, SwiGLU)',
                    'config':asdict(self.core.model.config),
                    'parametros':self.core.parameter_count,
                    'passos_de_treino':self.core.steps,
                    'exemplos_ensinados':self.memory.lesson_count(),
                    'premissas_de_raciocinio':self.memory.reasoning_fact_count(),
                    'palavras_iniciais':len(self.vocabulary),
                    'testes_independentes':len(self.memory.tests(10000)),
                    'autonomia':asdict(self.autonomy.settings),
                    'eventos_autonomos':self.memory.recent_events(5),
                    'dados':str(self.data_dir)}

    def close(self):
        self.autonomy.stop()
        with self.lock:
            self.core.save()
            self.memory.close()
