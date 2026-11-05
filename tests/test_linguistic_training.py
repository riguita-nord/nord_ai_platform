"""Regressões do treino linguístico misto PT-PT e do estado neuronal."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nord_brain.cli import handle
from nord_brain.core import NeuralCore, TransformerConfig
from nord_brain.linguistic_training import (
    TRAIN, HOLDOUT, _approved_corpus, evaluate, train,
)
from nord_brain.runtime import Brain

CONFIG = TransformerConfig(d_model=32, layers=2, heads=4, kv_heads=2,
                           intermediate_size=64, context=128)


class LinguisticTrainingTests(unittest.TestCase):
    def make_brain(self, root):
        brain = Brain(Path(root), start_autonomy=False)
        brain.core = NeuralCore(Path(root) / 'brain-v2.pt', CONFIG)
        return brain

    def test_training_and_holdout_are_separate_and_natural(self):
        questions = {question.casefold() for question, _ in TRAIN}
        reserved = {question.casefold() for question, _ in HOLDOUT}
        self.assertFalse(questions & reserved)
        self.assertGreaterEqual(len(TRAIN), 30)
        self.assertGreaterEqual(len(HOLDOUT), 10)
        self.assertTrue(any('Tu:' in q and 'Nord:' in q for q, _ in TRAIN))
        self.assertTrue(any('construir' in q for q, _ in TRAIN))
        text = _approved_corpus(None)
        self.assertIn('pequeno-almoço', text)
        self.assertIn('muda de direção', text)
        self.assertGreater(len(text.encode('utf-8')), 5000)

    def test_updates_real_transformer_without_adding_teacher_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = self.make_brain(directory)
            try:
                initial = evaluate(brain)
                self.assertEqual(initial['heldout'], len(HOLDOUT))
                steps = brain.core.steps
                result = train(brain, updates=2, rate=.000001)
                self.assertTrue(result['accepted'], result)
                self.assertGreater(result['steps'], 0)
                self.assertGreater(brain.core.steps, steps)
                self.assertEqual(brain.memory.teacher_lesson_count(), 0)
                self.assertEqual(brain.memory.get_meta('language_curriculum_cursor'), '2')
                self.assertEqual(evaluate(brain)['heldout'], initial['heldout'])
                self.assertEqual(
                    NeuralCore(Path(directory) / 'brain-v2.pt').steps,
                    brain.core.steps)
            finally:
                brain.close()

    def test_regression_restores_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = self.make_brain(directory)
            try:
                initial_steps = brain.core.steps
                # Four non-empty groups before and after; teacher and older
                # reference sets are empty on a clean installation.
                with patch.object(brain.core, 'score',
                                  side_effect=[5., 5., 5., 5., 6., 5., 5., 5.]):
                    result = train(brain, updates=1, rate=.000001)
                self.assertFalse(result['accepted'])
                self.assertEqual(brain.core.steps, initial_steps)
                self.assertIsNone(brain.memory.get_meta('language_curriculum_cursor'))
                self.assertEqual(
                    NeuralCore(Path(directory) / 'brain-v2.pt').steps,
                    initial_steps)
            finally:
                brain.close()

    def test_terminal_commands_and_explicit_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = self.make_brain(directory)
            approved = Path(directory) / 'minhas_frases.txt'
            approved.write_text(
                ('Hoje escrevi uma descrição de uma praça e de uma árvore. '
                 'Depois revi as frases para melhorar o português.\n') * 4,
                encoding='utf-8')
            try:
                self.assertIn('frases novas', handle(brain, '/avaliar-linguagem'))
                self.assertEqual(_approved_corpus(approved),
                                 approved.read_text(encoding='utf-8'))
                self.assertRaises(ValueError, train, brain, updates=49)
                self.assertRaises(ValueError, _approved_corpus,
                                  Path(directory) / 'nao_existe.txt')
                self.assertRaises(ValueError, handle, brain,
                                  '/treinar-linguagem 1 dois.txt extra')
            finally:
                brain.close()


if __name__ == '__main__':
    unittest.main()
