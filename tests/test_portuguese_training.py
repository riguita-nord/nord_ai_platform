"""Regression checks for CPU Portuguese dialogue training."""
import tempfile
import unittest
from pathlib import Path

from nord_brain.portuguese_training import curriculum,evaluate,train,describe
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore,TransformerConfig
from nord_brain.cli import handle
from nord_brain.autonomy import AutoSettings,Autonomy

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)

class PortugueseTrainingTests(unittest.TestCase):
    def prepare(self,directory):
        brain=Brain(Path(directory),start_autonomy=False)
        brain.core=NeuralCore(Path(directory)/'brain-v2.pt',CONFIG)
        return brain

    def test_curriculum_disjoint(self):
        training,heldout=curriculum()
        self.assertEqual(len(training),54)
        self.assertEqual(len(heldout),18)
        self.assertFalse({q for q,_ in training}&{q for q,_ in heldout})
        self.assertTrue(any('história' in q for q,_ in training))
        self.assertTrue(any('opinião' in q for q,_ in training))

    def test_cpu_gradient_training_does_not_create_teacher_facts(self):
        with tempfile.TemporaryDirectory() as folder:
            brain=self.prepare(folder)
            try:
                before=evaluate(brain)
                steps=brain.core.steps
                result=train(brain,examples=3)
                self.assertTrue(result.accepted,describe(result))
                self.assertGreater(brain.core.steps,steps)
                self.assertEqual(brain.memory.teacher_lesson_count(),0)
                self.assertIsNone(brain.memory.exact('Podes discordar de mim?'))
                self.assertEqual(evaluate(brain)['examples'],before['examples'])
                self.assertEqual(brain.memory.get_meta('portuguese_curriculum_cursor'),'3')
            finally:brain.close()

    def test_teacher_pause_respected(self):
        with tempfile.TemporaryDirectory() as folder:
            brain=self.prepare(folder)
            try:
                brain.memory.pause_autonomy(True)
                self.assertRaises(ValueError,train,brain,2)
            finally:brain.close()

    def test_cli_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            brain=self.prepare(folder)
            try:
                self.assertIn('Perda em português:',handle(brain,'/avaliar-portugues'))
                self.assertIn('Exemplos: 2',handle(brain,'/treinar-portugues 2'))
                self.assertRaises(ValueError,handle,brain,'/treinar-portugues 101')
            finally:brain.close()

    def test_autonomy_runs_only_on_third_tick(self):
        with tempfile.TemporaryDirectory() as folder:
            brain=self.prepare(folder)
            try:
                agent=Autonomy(brain,AutoSettings(training=True,experiments=False,
                    world_simulations=False,cognitive_lab=False,research=False,
                    growth=False,portuguese_practice=True))
                agent.tick()
                agent.tick()
                self.assertEqual(brain.memory.get_meta('portuguese_curriculum_cursor'),None)
                agent.tick()
                self.assertEqual(brain.memory.get_meta('portuguese_curriculum_cursor'),'1')
            finally:brain.close()

if __name__=='__main__':unittest.main()
