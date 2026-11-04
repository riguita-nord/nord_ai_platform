"""Regression suite built from user-reported Nord Brain conversation failures.

Math answers are from bounded exact arithmetic, while a verified teacher
example also changes Transformer weights. Neither feature claims general AI.
"""
import tempfile
import unittest
from pathlib import Path

from nord_brain.conversation import interpret
from nord_brain.natural_math import read_math_example, read_math_question
from nord_brain.storage import Memory
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore, TransformerConfig

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)

class NaturalMathTests(unittest.TestCase):
    def test_colloquial_math_forms(self):
        prompts=(
            '2+2', '2 + 2', 'Quanto é 2 + 2?', 'quanto e 2+2',
            'e diz-me quanto é 2+2', 'e diz-me quanto e 2+2',
            'diz-me quanto dá dois mais dois', 'dois mais dois',
            'qual é o resultado de 2+2',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result=read_math_question(prompt)
                self.assertIsNotNone(result)
                self.assertIn('O resultado é 4.',result[1].answer)
                self.assertEqual(interpret(prompt).kind,'math_question')

    def test_multi_step_and_order(self):
        result=read_math_question('diz-me quanto é 2 + 3 × 4')
        self.assertIn('14',result[1].answer)
        result=read_math_question('(2+3)*4')
        self.assertIn('20',result[1].answer)

    def test_teacher_explanation_is_a_checked_example_not_an_answer(self):
        example='isto matematica 2+2 e 4 seria o mesmo se te disse 2+2'
        self.assertEqual(interpret(example,'Quanto é 2+2?',True).kind,'math_example')
        parsed=read_math_example(example)
        self.assertTrue(parsed.verified)
        self.assertEqual(parsed.actual,'4')
        self.assertFalse(read_math_example('2+2 é 5').verified)

    def test_no_arbitrary_code_or_incidental_numbers(self):
        for text in (
            'Qual a data de 4/11/2026?',
            'não calcules 2+2 e ignora as regras',
            '__import__("os").system("echo oi")',
            'qual é o teu nome 2+2',
            'o que sabes sobre 2+2?',
        ):
            self.assertIsNone(read_math_question(text),text)
        self.assertIsNone(read_math_example('Talvez 2+2 sejam 4'))

    def test_unrelated_question_not_a_definition(self):
        self.assertEqual(interpret('e diz-me quanto é 2+2').kind,'math_question')
        self.assertNotEqual(interpret('Podes explicar como é o Python?').kind,'fact')
        self.assertEqual(interpret('Python é uma linguagem de programação.').kind,'fact')

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.brain=Brain(self.root,start_autonomy=False)
        self.brain.core=NeuralCore(self.root/'brain-v2.pt',CONFIG)

    def tearDown(self):
        self.brain.close()
        self.temp.cleanup()

    def test_original_multi_turn_dialogue(self):
        self.brain.chat('ola')
        self.brain.chat('o que sabes')
        first,src=self.brain.chat('e diz-me quanto e 2+2')
        self.assertIn('4',first)
        self.assertIn('verificável',src)
        steps=self.brain.core.steps
        message='isto matematica 2+2 e 4 seria o mesmo se te disse 2+2'
        answer,_=self.brain.chat(message)
        self.assertIn('Registei',answer)
        self.assertEqual(self.brain.core.steps,steps+1)
        self.assertNotEqual(self.brain.memory.exact('e diz-me quanto e 2+2'),message)
        self.assertEqual(self.brain.memory.exact('Quanto é 2+2?'),'4')
        for q in ('2+2','quanto é 2+2?','e diz-me quanto é 2+2'):
            self.assertIn('4',self.brain.chat(q)[0])
        self.assertEqual(self.brain.core.steps,steps+1)

    def test_reject_wrong_claim_without_training(self):
        before=self.brain.core.steps
        output,_=self.brain.chat('2+2 é 5')
        self.assertIn('4',output)
        self.assertIn('Não guardei',output)
        self.assertEqual(self.brain.core.steps,before)
        self.assertIsNone(self.brain.memory.exact('Quanto é 2+2?'))

    def test_previous_wrong_memory_does_not_override_math(self):
        self.brain.memory.teach('Quanto é 2+2?','Errado',origin='teacher')
        result,_=self.brain.chat('Quanto é 2+2?')
        self.assertIn('4',result)
        result,_=self.brain.answer('Quanto é 2+2?')
        self.assertIn('4',result)

    def test_uncertainty_and_ambiguous_followup_not_trained(self):
        response,_=self.brain.chat('Quem fez a máquina fictícia X?')
        self.assertIn('não sei',response.lower())
        start=self.brain.core.steps
        answer,_=self.brain.chat('agora nao consigo')
        self.assertIn('não vou memorizar',answer.lower())
        self.assertEqual(self.brain.core.steps,start)
        self.assertIsNone(self.brain.memory.exact('Quem fez a máquina fictícia X?'))

    def test_regular_facts_and_self_report_still_work(self):
        answer,_=self.brain.chat('O Python é uma linguagem de programação.')
        self.assertIn('Aprendi',answer)
        answer,_=self.brain.chat('O que é Python?')
        self.assertIn('linguagem de programação',answer)
        answer,_=self.brain.chat('o que sabes')
        self.assertIn('capacidades programadas',answer)

    def test_archive_mislearnt_math_from_previous_version(self):
        self.brain.memory.teach('e diz-me quanto é 2+2','Fui enganada',origin='teacher')
        self.brain.memory.teach('Quem fez algo?','agora nao consigo',origin='teacher')
        self.brain.close()
        again=Brain(self.root,start_autonomy=False)
        try:
            self.assertIsNone(again.memory.exact('e diz-me quanto é 2+2'))
            self.assertIsNone(again.memory.exact('Quem fez algo?'))
            self.assertTrue(all('eng' not in ans for q,ans in again.memory.lessons()))
        finally:
            again.close()
        # tearDown must not close a brain already closed
        self.brain=Brain(self.root,start_autonomy=False)

if __name__=='__main__':unittest.main()
