"""Regression tests for conversational intent and grounded self-knowledge.

Tests distinguish symbolic interpretation from online neural learning.
"""
import tempfile
import unittest
from pathlib import Path

from nord_brain.conversation import interpret, normalize_request
from nord_brain.storage import Memory
from nord_brain.self_knowledge import report
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore, TransformerConfig

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)


class IntentTests(unittest.TestCase):
    def test_common_self_questions_and_accents(self):
        for q in ('o que sabes','O que sabes?','O QUE SABES!',
                  'O que sabes fazer?', 'Quais são as tuas capacidades?',
                  'O que aprendeste?','O que já aprendeste?'):
            self.assertEqual(interpret(q, 'uma pergunta antiga',True).kind,
                             'self_knowledge',q)
        self.assertEqual(normalize_request('O QUE SABES?!'),'o que sabes')

    def test_meta_instruction_is_not_literal_answer(self):
        phrase='quando te digo o que sabes eu quero que me digas o que sabes fazer e o que aprendeste'
        intent=interpret(phrase,'o que sabes',True)
        self.assertEqual(intent.kind,'intent_instruction')
        self.assertEqual(intent.topic,'self_knowledge')
        self.assertEqual(intent.question,'o que sabes')
        self.assertEqual(intent.answer,'me digas o que sabes fazer e o que aprendeste')

    def test_custom_alias_instruction(self):
        q='Quando te digo "faz um balanço", quero que me digas as tuas capacidades e o que aprendeste.'
        intent=interpret(q,'uma pergunta',True)
        self.assertEqual(intent.kind,'intent_instruction')
        self.assertEqual(intent.question,'faz um balanço')
        self.assertEqual(intent.topic,'self_knowledge')

    def test_unsupported_rule_not_taught_as_answer(self):
        intent=interpret('quando te digo azul eu quero que cantes uma música','azul',True)
        self.assertEqual(intent.kind,'intent_instruction')
        self.assertEqual(intent.topic,'unsupported')

    def test_topic_question_is_not_about_brain(self):
        self.assertEqual(interpret('O que sabes sobre a Lua?').kind,'question')


class StorageTests(unittest.TestCase):
    def test_alias_and_teacher_items_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'memory.sqlite3'
            memory=Memory(path)
            memory.set_intent_alias('faz o meu balanço','self_knowledge')
            memory.concept('python','python é uma linguagem.')
            memory.teach('Quanto é 2+2?','4')
            self.assertEqual(memory.teacher_lesson_count(),1)
            self.assertTrue(any('python' in s for s in memory.recent_teacher_learnings()))
            memory.close()
            memory=Memory(path)
            self.assertEqual(memory.resolve_intent_alias('FAZ O MEU BALANÇO?'),'self_knowledge')
            self.assertIsNone(memory.resolve_intent_alias('outra coisa'))
            memory.close()

    def test_legacy_misunderstanding_is_archived_not_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'memory.sqlite3'
            memory=Memory(path)
            memory.teach('o que sabes','quando te digo o que sabes eu quero que me digas o que sabes fazer')
            memory.teach('quanto é 2+2?','4')
            memory.close()
            memory=Memory(path)
            self.assertIsNone(memory.exact('o que sabes'))
            self.assertEqual(memory.exact('quanto é 2+2?'),'4')
            self.assertEqual(memory.lesson_count(),1)
            self.assertEqual(memory.teacher_lesson_count(),1)
            self.assertFalse(any(q=='o que sabes' for q,_ in memory.lessons()))
            count=memory.db.execute(
                "SELECT COUNT(*) FROM lessons WHERE origin='legacy_misinterpreted_instruction'"
            ).fetchone()[0]
            self.assertEqual(count,1)
            memory.close()


class ConversationTests(unittest.TestCase):
    def make_brain(self,folder):
        brain=Brain(Path(folder),start_autonomy=False)
        brain.core=NeuralCore(Path(folder)/'brain-v2.pt',CONFIG)
        return brain

    def test_original_reproduction_and_dynamic_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=self.make_brain(tmp)
            try:
                reply,_=brain.chat('o que sabes')
                self.assertIn('capacidades programadas',reply)
                self.assertNotIn('quando te digo o que sabes',reply)
                steps=brain.core.steps
                explanation='quando te digo o que sabes eu quero que me digas o que sabes fazer e o que aprendeste'
                understood,source=brain.chat(explanation)
                self.assertIn('Percebi',understood)
                self.assertIn('interpretação guardada',source)
                self.assertEqual(brain.core.steps,steps)
                self.assertIsNone(brain.memory.exact('o que sabes'))
                again,_=brain.chat('o que sabes')
                self.assertIn('passos de treino neuronal',again)
                self.assertNotIn(explanation,again)
            finally:
                brain.close()

    def test_reports_new_teacher_concepts_not_stale_canned_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=self.make_brain(tmp)
            try:
                before,_=brain.chat('Quais são as tuas capacidades?')
                brain.chat('Python é uma linguagem de programação.')
                after,_=brain.chat('O que sabes fazer?')
                self.assertIn('python',after.lower())
                self.assertIn('linguagem de programação',after)
                self.assertNotIn('linguagem de programação',before)
                self.assertGreater(brain.core.steps,0)
            finally:
                brain.close()

    def test_custom_phrase_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=self.make_brain(tmp)
            try:
                brain.chat('Quando te digo faz o meu balanço, quero que me digas o que sabes fazer e o que aprendeste.')
                answer,_=brain.chat('faz o meu balanço')
                self.assertIn('capacidades programadas',answer)
            finally:
                brain.close()
            again=Brain(Path(tmp),start_autonomy=False)
            try:
                answer,_=again.chat('FAZ O MEU BALANÇO?')
                self.assertIn('capacidades programadas',answer)
            finally:
                again.close()

    def test_unsupported_rule_does_not_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=self.make_brain(tmp)
            try:
                before=brain.core.steps
                answer,_=brain.chat('quando te digo azul eu quero que cantes uma música')
                self.assertIn('ainda só consigo',answer)
                self.assertEqual(brain.core.steps,before)
                self.assertIsNone(brain.memory.resolve_intent_alias('azul'))
            finally:
                brain.close()


if __name__=='__main__':unittest.main()
