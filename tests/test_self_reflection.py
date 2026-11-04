"""Regression tests for grounded self-knowledge, routing, and memory safety."""
from pathlib import Path
import tempfile
import unittest
from nord_brain.conversation import interpret, self_query_kind
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore, TransformerConfig

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)


class SelfKindTests(unittest.TestCase):
    def test_self_queries(self):
        expected = {
            'summary': ('O que sabes?', 'Conta-me tudo sobre ti', 'O que sabes sobre ti?'),
            'identity': ('Quem és tu?', 'Como te chamas?'),
            'creator': ('Quem te criou?',),
            'version': ('Qual é a tua versão?',),
            'age': ('Quando nasceste?',),
            'location': ('Onde estás a correr?',),
            'purpose': ('Para que serves?',),
            'limitations': ('Quais são os teus limites?',),
            'architecture': ('Como funcionas?', 'Usas Qwen?'),
            'size': ('Quantos parâmetros tens?', 'Quantos neurónios tens?'),
            'memory': ('Tens memória?', 'Lembras-te do que te disse?'),
            'learned': ('O que aprendeste recentemente?',),
            'training': ('Estás a aprender?',),
            'internet': ('Tens internet?',),
            'experiments': ('Tens um mundo virtual?',),
            'experience': ('Tens sentimentos?', 'És consciente?'),
        }
        for kind, questions in expected.items():
            for question in questions:
                with self.subTest(question=question):
                    self.assertEqual(self_query_kind(question), kind)
                    self.assertEqual(interpret(question,'pergunta antiga',True).kind,'self_knowledge')

    def test_external_queries_and_arithmetic_still_separate(self):
        for question in ('O que sabes sobre Python?', 'O que sabes sobre 2+2?',
                         'Quem é Elon Musk?', 'Qual a versão do QBCore?',
                         'Quanto é 2+2?', 'Tenho 2-3 jogadores',
                         'O teu servidor tem 2 jogadores?', 'Tens 10 + 2?',
                         'Pausa aprendizagem'):
            with self.subTest(question=question):
                self.assertIsNone(self_query_kind(question))


class RuntimeSelfKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.brain=Brain(Path(self.temp.name),start_autonomy=False)
        self.brain.core=NeuralCore(Path(self.temp.name)/'brain-v2.pt',CONFIG)

    def tearDown(self):
        self.brain.close()
        self.temp.cleanup()

    def test_real_introspection_without_training(self):
        count=self.brain.core.steps
        expected={
            'Quem és tu?': 'Nord Brain',
            'Qual é a tua versão?': '1.2.0',
            'Quantos parâmetros tens?': str(self.brain.core.parameter_count),
            'Quantas camadas tens?': '2 camadas',
            'Como funcionas?': 'Transformer',
            'Tens memória?': 'memory.sqlite3',
            'Estás a aprender?': 'passos de treino',
            'Tens internet?': 'pesquisa',
            'Quais são os teus limites?': 'ainda não compreendo',
            'Quem te criou?': 'Nord Lab',
        }
        for q, expected_text in expected.items():
            answer,source=self.brain.chat(q)
            self.assertIn(expected_text.lower(),answer.lower(),q)
            self.assertIn('autoconhecimento',source,q)
        self.assertEqual(count,self.brain.core.steps)
        self.assertIsNone(self.brain.last_question)
        self.assertFalse(self.brain.awaiting_explanation)

    def test_memory_report_changes_with_actual_learning(self):
        before,_=self.brain.chat('O que tens na memória?')
        self.assertNotIn('linguagem de programação',before)
        self.brain.chat('Python é uma linguagem de programação.')
        after,_=self.brain.chat('O que tens na memória?')
        self.assertIn('linguagem de programação',after)
        report,_=self.brain.chat('O que sabes?')
        self.assertIn('capacidades programadas',report)


if __name__ == '__main__':
    unittest.main()
