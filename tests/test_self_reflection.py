"""Grounded self-introspection with ordinary Portuguese phrasing."""
from pathlib import Path
import tempfile
import unittest
from nord_brain.conversation import interpret, self_query_kind
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore, TransformerConfig

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)

class SelfKindTests(unittest.TestCase):
    def test_broad_self_inquiries(self):
        examples={
            'summary':['o que sabes','Conta-me tudo sobre ti','o que sabes sobre ti?'],
            'identity':['Quem és tu?','Como te chamas?', 'És uma IA?'],
            'creator':['Quem te criou?','Quem te programou?'],
            'version':['Qual é a tua versão?','Estás atualizada?'],
            'age':['Quando nasceste?','Quantos anos tens?'],
            'location':['Onde estás a correr?'],
            'purpose':['Para que serves?','Qual é o teu objetivo?'],
            'limitations':['O que não sabes?','Quais são os teus limites?'],
            'architecture':['Como funcionas?','Usas Qwen?','Como raciocinas?'],
            'size':['Quantos parâmetros tens?','Quantos neurónios tens?'],
            'memory':['Tens memória?','Lembras-te do que te disse?'],
            'learned':['O que aprendeste recentemente?'],
            'training':['Estás a aprender?','Quantos passos de treino tens?'],
            'internet':['Tens internet?','De onde vem o teu conhecimento?'],
            'experiments':['Tens um mundo virtual?','Quantas simulações fizeste?'],
            'experience':['Tens sentimentos?','És consciente?']
        }
        for expected,phrases in examples.items():
            for q in phrases:
                with self.subTest(q=q):
                    self.assertEqual(self_query_kind(q),expected)
                    self.assertEqual(interpret(q,last_question='outra coisa',awaiting_explanation=True).kind,'self_knowledge')

    def test_other_subjects_and_mathematics_are_not_self_questions(self):
        for q in ('O que sabes sobre Python?', 'O que sabes sobre 2+2?',
                  'Quem é o Elon Musk?', 'Qual a versão do QBCore?',
                  'Quanto é 2+2?', 'Tenho 2-3 jogadores',
                  'O teu servidor tem 2 jogadores?', 'Tens 10 + 2?',
                  'Pausa aprendizagem', 'Experimente matemática'):
            with self.subTest(q=q):
                self.assertIsNone(self_query_kind(q))

class SelfRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.brain=Brain(Path(self.temp.name),start_autonomy=False)
        self.brain.core=NeuralCore(Path(self.temp.name)/'brain-v2.pt',CONFIG)

    def tearDown(self):
        self.brain.close();self.temp.cleanup()

    def test_answers_are_grounded_and_do_not_train(self):
        start=self.brain.core.steps
        from nord_brain import __version__
        cases=[('Quem és tu?','Nord Brain'),('Qual é a tua versão?',__version__),
               ('Quantos parâmetros tens?',str(self.brain.core.parameter_count)),
               ('Quantas camadas tens?','2 camadas'),
               ('Como funcionas?','Transformer'),
               ('Tens memória?','memory.sqlite3'),
               ('Estás a aprender?','passos de treino'),
               ('Tens internet?','pesquisa'),
               ('Quais são os teus limites?','ainda não compreendo'),
               ('Quando nasceste?','não tenho uma data'),
               ('Quem te criou?','Nord Lab'),
               ('Tens sentimentos?','programa de IA')]
        for question,expected in cases:
            answer,origin=self.brain.chat(question)
            self.assertIn(expected.lower(),answer.lower(),question)
            self.assertIn('autoconhecimento',origin,question)
        self.assertEqual(self.brain.core.steps,start)
        self.assertIsNone(self.brain.last_question)
        self.assertFalse(self.brain.awaiting_explanation)

    def test_real_memories_are_reflected_and_not_overwritten(self):
        before,_=self.brain.chat('O que tens na memória?')
        self.assertNotIn('Python é uma linguagem',before)
        self.brain.chat('Python é uma linguagem de programação.')
        after,_=self.brain.chat('O que tens na memória?')
        self.assertIn('linguagem de programação',after)
        report,_=self.brain.chat('O que sabes?')
        self.assertIn('capacidades programadas',report)

    def test_unknown_personal_attribute_is_not_hallucinated(self):
        for question in ('Qual é a tua cor favorita?', 'Tens irmãos?', 'Como te sentes?'):
            reply,source=self.brain.chat(question)
            self.assertIn('não tenho informação verificada',reply.lower())
            self.assertIn('autoconhecimento',source)

    def test_not_the_same_as_research_other_people(self):
        self.assertIsNone(self_query_kind('Quem é Elon Musk?'))
        self.assertIsNone(self_query_kind('O que sabes sobre o Java?'))
        self.assertIsNone(self_query_kind('Qual é a tua versão do script?'))

if __name__=='__main__':unittest.main()
