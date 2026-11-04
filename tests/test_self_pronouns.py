"""Regression checks for second-person references to Nord Brain.

These are bounded dialogue-intent rules and actual runtime reports; they are
not evidence of arbitrary Portuguese language comprehension by the Transformer.
"""
from pathlib import Path
import tempfile
import unittest

from nord_brain.conversation import self_query_kind, interpret
from nord_brain.storage import Memory
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore, TransformerConfig


CONFIG=TransformerConfig(
    d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128
)


class PronounTests(unittest.TestCase):
    def test_equivalent_phrasings_address_the_ai(self):
        examples={
            'o que tu sabes':'summary',
            'o que tu sabes fazer':'summary',
            'e o que tu sabes fazer':'summary',
            'e o mesmo que te disser o que tu sabes':'summary',
            'quem és tu?':'identity',
            'tu és a Nord Brain?':'identity',
            'tu tens memória?':'memory',
            'o que tu sabes sobre a tua memória?':'memory',
            'tu tens internet?':'internet',
            'como funciona o teu cérebro?':'architecture',
            'tu tens sentimentos?':'experience',
        }
        for question,kind in examples.items():
            with self.subTest(question=question):
                self.assertEqual(self_query_kind(question),kind)
                self.assertEqual(interpret(question,'o que tu sabes',True).kind,
                                 'self_knowledge')

    def test_external_questions_remain_external(self):
        for phrase in (
            'quem é Elon Musk?',
            'o que tu sabes sobre a Lua?',
            'tu sabes quem é Elon Musk?',
            'o meu nome é Ricardo.',
            'tens 10 + 2?',
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(self_query_kind(phrase))


class MemoryRegressionTests(unittest.TestCase):
    def test_old_paraphrase_is_archived_but_normal_teaching_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'memory.sqlite3'
            memory=Memory(path)
            memory.teach('o que tu sabes','e o que tu sabes fazer')
            memory.teach('o que sabes','quando te digo o que sabes eu quero que me digas tudo')
            memory.teach('Quanto é 2+2?','4')
            memory.close()

            restored=Memory(path)
            try:
                self.assertIsNone(restored.exact('o que tu sabes'))
                self.assertEqual(restored.exact('Quanto é 2+2?'),'4')
                self.assertEqual(restored.teacher_lesson_count(),1)
                self.assertFalse(any('o que tu sabes' in q for q,_ in restored.lessons()))
            finally:
                restored.close()

    def test_question_about_ai_is_live_not_a_memory_echo(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=Brain(Path(tmp),start_autonomy=False)
            brain.core=NeuralCore(Path(tmp)/'brain-v2.pt',CONFIG)
            try:
                before=brain.core.steps
                for question in ('o que tu sabes','o que tu sabes fazer',
                                 'e o que tu sabes fazer',
                                 'e o mesmo que te disser o que tu sabes'):
                    answer,origin=brain.chat(question)
                    self.assertIn('capacidades programadas',answer.lower())
                    self.assertIn('autoconhecimento',origin)
                self.assertEqual(brain.core.steps,before)
            finally:
                brain.close()


if __name__=='__main__':
    unittest.main()
