"""Generalisation must be evaluated using unseen neural predictions.

The deterministic verifier is allowed to answer safely, but cannot earn a
model competence score. Training targets are procedurally generated.
"""
import tempfile
import unittest
from pathlib import Path

from nord_brain.autonomy import Autonomy, AutoSettings
from nord_brain.core import NeuralCore, TransformerConfig
from nord_brain.experiments import MAX_LEVEL, curriculum
from nord_brain.reasoning import math_question
from nord_brain.runtime import Brain

SMALL=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)


class CurriculumTests(unittest.TestCase):
    def test_no_answers_or_question_pairs_hardcoded(self):
        training, tests=curriculum(0)
        self.assertGreater(len(training),70)
        self.assertGreater(len(tests),6)
        self.assertTrue({q for q,a in training}.isdisjoint({q for q,a in tests}))
        # A change in operands still belongs to the task family, but the
        # particular held-out pair is not trained or remembered as a lesson.
        self.assertIn(('Quanto é 2 + 2?','4'),training)
        self.assertIn(('Quanto é 4 + 4?','8'),tests)
        for level in range(MAX_LEVEL+1):
            training,tests=curriculum(level)
            self.assertTrue({q for q,a in training}.isdisjoint({q for q,a in tests}))
            for question,answer in (training[:30]+tests[:30]):
                self.assertIn('O resultado é '+answer+'.', math_question(question).answer)

    def test_determinism_and_increasing_scope(self):
        self.assertEqual(curriculum(0),curriculum(0))
        self.assertTrue(any('99' in q for q,a in curriculum(2)[0]))
        self.assertTrue(any('×' in q for q,a in curriculum(3)[0]))


class NeuralGeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.brain=Brain(self.root,start_autonomy=False)
        self.brain.core=NeuralCore(self.root/'brain-v2.pt',SMALL)

    def tearDown(self):
        self.brain.close()
        self.temp.cleanup()

    def test_math_answers_use_a_verifier_but_credit_the_network_only_if_correct(self):
        self.brain.core.generate=lambda *a,**k: 'estou errado'
        answer,origin=self.brain.chat('4+4')
        self.assertIn('8',answer)
        self.assertIn('tentativa neuronal ainda incorreta',origin)
        answer2,origin2=self.brain.chat('e diz-me quanto é 4+4')
        self.assertIn('8',answer2)
        self.assertIn('incorreta',origin2)
        self.brain.core.generate=lambda *a,**k: '8'
        answer3,origin3=self.brain.chat('quanto é 4 + 4?')
        self.assertIn('8',answer3)
        self.assertIn('neuronal correta',origin3)
        self.assertEqual(self.brain.core.steps,0)  # A query is not a teaching label.

    def test_training_experiences_modify_weights_without_leaking_test_inputs(self):
        agent=Autonomy(self.brain,AutoSettings(training=True,experiments=True,
                     growth=False,research=False,examples_per_tick=4))
        before=self.brain.core.steps
        agent.tick()
        self.assertEqual(self.brain.core.steps,before+5)  # 4 new + one replay.
        self.assertGreaterEqual(len(self.brain.memory.tests(1000)),7)
        self.assertEqual(self.brain.memory.exact('Quanto é 4 + 4?'),None)
        self.assertEqual(self.brain.memory.get_meta('curriculum_cursor'),'4')

    def test_auto_level_up_requires_neural_correctness_on_holdouts(self):
        agent=Autonomy(self.brain,AutoSettings(training=True,experiments=True,
                     growth=False,research=False,examples_per_tick=1))
        self.brain.core.generate=lambda *a,**k: 'completamente errado'
        for _ in range(4):agent.tick()
        self.assertEqual(self.brain.memory.get_meta('math_level','0'),'0')
        self.assertEqual(self.brain.memory.get_meta('math_neural_accuracy'),'0/8')
        answer_map=dict(curriculum(0)[1])
        self.brain.core.generate=lambda q,**k: answer_map.get(q,'?')
        for _ in range(4):agent.tick()
        self.assertEqual(self.brain.memory.get_meta('math_level'),'1')
        self.assertEqual(self.brain.memory.get_meta('math_neural_accuracy'),'8/8')
        self.assertTrue(any(e[0]=='math_level_up' for e in self.brain.memory.recent_events(15)))


if __name__=='__main__':unittest.main()
