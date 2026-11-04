"""Coordinates local teaching, neural weights and sourced memory."""
from pathlib import Path
from dataclasses import asdict
import os
from .core import NeuralCore, TransformerConfig
from .storage import Memory
from .vocabulary import initialize


class Brain:
    def __init__(self, data_dir: Path):
        self.data_dir=Path(data_dir)
        self.data_dir.mkdir(parents=True,exist_ok=True)
        self.memory=Memory(self.data_dir/'memory.sqlite3')
        self.vocabulary=initialize(self.data_dir/'vocabulary.json')
        profiles = {
            'tiny': TransformerConfig(d_model=64,layers=2,heads=2,kv_heads=1,intermediate_size=192,context=128),
            'small': TransformerConfig(),
            'medium': TransformerConfig(d_model=192,layers=6,heads=6,kv_heads=3,intermediate_size=512,context=384),
        }
        profile = os.environ.get('NORD_BRAIN_PRESET','small').lower()
        if profile not in profiles:
            raise ValueError('NORD_BRAIN_PRESET deve ser tiny, small ou medium')
        # A saved checkpoint always wins over the preset to preserve previous training.
        self.core=NeuralCore(self.data_dir/'brain-v2.pt',config=profiles[profile])
        self.last_question=None

    def teach(self,question,answer,epochs=2):
        if not 1 <= len(question.strip()) <= 250 or not 1 <= len(answer.strip()) <= 350:
            raise ValueError('Exemplo inválido')
        if self.memory.is_holdout(question):
            raise ValueError('Pergunta reservada para avaliação: não pode ser ensinada')
        # Save the last valid checkpoint before training, then persist the updated weights.
        self.core.save()
        try:
            loss=self.core.train(question,answer,epochs=epochs)
            self.core.save()
        except Exception:
            self.core.load()
            raise
        self.memory.teach(question,answer)
        self.memory.metric('train_loss',loss)
        return loss

    def answer(self,question,neural_only=False):
        question=question.strip()
        self.last_question=question
        if not neural_only:
            known=self.memory.exact(question)
            if known is not None:
                return known,'memória (exemplo ensinado, não generalização neuronal)'
            concept=self.memory.lookup_concept(question)
            if concept:
                return concept,'dicionário de conceitos'
        if self.memory.lesson_count()<15 and not neural_only:
            return 'Ainda não consigo responder. Ensina-me com /ensinar pergunta => resposta.','sem conhecimento neuronal suficiente'
        prediction=self.core.generate(question)
        return (prediction or 'Não consegui gerar uma resposta.'),'rede neuronal experimental (não verificado)'

    def practice(self,limit=20):
        examples=self.memory.lessons(limit)
        if not examples:return 0,None
        losses=[self.core.train(q,a,epochs=1) for q,a in reversed(examples)]
        self.core.save()
        value=sum(losses)/len(losses)
        self.memory.metric('practice_loss',value)
        return len(losses),value

    def evaluate(self):
        tests=self.memory.tests(10)
        if not tests: return None
        correct=0
        for question,answer in tests:
            predicted=self.core.generate(question,temperature=0).strip().casefold()
            correct += predicted==answer.strip().casefold()
        score=correct/len(tests)
        self.memory.metric('heldout_accuracy',score)
        return correct,len(tests),score

    def grow(self,extra=16):
        self.core.grow(extra)
        self.core.save()
        return self.core.hidden_size

    def status(self):
        return {'largura_mlp':self.core.hidden_size,
                'arquitetura':'decoder-only Transformer (RoPE, GQA, RMSNorm, SwiGLU)',
                'config':asdict(self.core.model.config),
                'parametros':self.core.parameter_count,
                'passos_de_treino':self.core.steps,
                'exemplos_ensinados':self.memory.lesson_count(),
                'palavras_iniciais':len(self.vocabulary),
                'testes_independentes':len(self.memory.tests(10000)),
                'dados':str(self.data_dir)}

    def close(self):
        self.core.save()
        self.memory.close()
