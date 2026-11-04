"""The world brain must learn consequences from interaction, not memorise text.

These tests explicitly measure predictive generalisation to DIFFERENT virtual
world seeds, and show that the learned parameters persist across restarts.
"""
import tempfile
import unittest
from pathlib import Path

import torch

from nord_brain.world import ACTION_NAMES, GridWorld, WorldMind, encode
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore, TransformerConfig
from nord_brain.autonomy import Autonomy, AutoSettings

SMALL=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,
                        intermediate_size=64,context=128)


class VirtualWorldTests(unittest.TestCase):
    def test_observation_does_not_contain_full_map(self):
        env=GridWorld(55)
        sensed=env.sense()
        self.assertEqual(len(sensed),6)
        self.assertEqual(len(encode(sensed,2)),10)
        for action in range(4):
            self.assertIsInstance(env.move(action)[2],float)

    def test_hypotheses_are_four_distinct_actions(self):
        with tempfile.TemporaryDirectory() as d:
            mind=WorldMind(Path(d)/'world.pt')
            hypotheses=mind.hypothesize(GridWorld(8).sense())
            self.assertEqual([h['name'] for h in hypotheses],list(ACTION_NAMES))
            self.assertEqual({h['action'] for h in hypotheses},set(range(4)))
            for h in hypotheses:
                self.assertTrue(0 <= h['collision_risk'] <= 1)
                self.assertEqual(len(h['delta']),2)

    def test_learning_from_experience_on_unseen_worlds(self):
        with tempfile.TemporaryDirectory() as d:
            mind=WorldMind(Path(d)/'model.pt')
            before=mind.evaluate()['joint_loss']
            weights=[p.detach().clone() for p in mind.model.parameters()]
            for _ in range(7):
                mind.experience_cycle(scenarios=6,steps_per_world=16,
                                      updates=48,save=False)
            after=mind.evaluate()['joint_loss']
            self.assertGreater(mind.observations,160)
            self.assertGreater(mind.train_steps,300)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(weights,mind.model.parameters())))
            # Tests use distinct world seeds, never replayed as training episodes.
            self.assertLess(after,before*.30,(before,after))

    def test_weights_and_experiences_persist(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'
            brain=WorldMind(path)
            brain.experience_cycle(3,steps_per_world=12,updates=30)
            probe=GridWorld(100_001).sense()
            original=brain.hypothesize(probe)
            reopened=WorldMind(path)
            self.assertEqual(reopened.observations,brain.observations)
            self.assertEqual(reopened.train_steps,brain.train_steps)
            after=reopened.hypothesize(probe)
            for a,b in zip(original,after):
                self.assertAlmostEqual(a['collision_risk'],b['collision_risk'])
                self.assertAlmostEqual(a['score'],b['score'])

    def test_simulation_is_not_real_hardware_control(self):
        with tempfile.TemporaryDirectory() as d:
            mind=WorldMind(Path(d)/'world.pt')
            with self.assertRaises(ValueError):
                mind.experience_cycle(scenarios=100)
            with self.assertRaises(ValueError):
                GridWorld(1,size=100)
            result=mind.experience_cycle(scenarios=2,steps_per_world=5,updates=3)
            self.assertEqual(result['scenarios'],2)
            self.assertIn('colisões',mind.describe_cycle(result))


class IntegrationTests(unittest.TestCase):
    def _brain(self,folder):
        brain=Brain(Path(folder),start_autonomy=False)
        brain.core=NeuralCore(Path(folder)/'brain-v2.pt',SMALL)
        return brain

    def test_normal_conversation_can_trigger_real_simulation(self):
        with tempfile.TemporaryDirectory() as d:
            brain=self._brain(d)
            try:
                before=brain.world.train_steps
                response,origin=brain.chat('simula 2 cenários')
                self.assertIn('quatro hipóteses',response)
                self.assertIn('executei',response.lower())
                self.assertIn('mundo virtual',origin)
                self.assertGreater(brain.world.train_steps,before)
                self.assertEqual(brain.status()['modelo_de_mundo']['cenarios'],2)
            finally:brain.close()

    def test_autonomous_cycle_learns_without_teacher_command(self):
        with tempfile.TemporaryDirectory() as d:
            brain=self._brain(d)
            try:
                agent=Autonomy(brain,AutoSettings(training=True,experiments=False,
                    growth=False,research=False,world_simulations=True,world_scenarios=3))
                agent.tick()
                self.assertGreater(brain.world.observations,0)
                self.assertGreater(brain.world.train_steps,0)
                self.assertTrue(any(kind=='world_training' for kind,_,_ in
                                    brain.memory.recent_events(8)))
            finally:brain.close()

    def test_world_simulation_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            brain=self._brain(d)
            try:
                agent=Autonomy(brain,AutoSettings(training=True,experiments=False,
                    growth=False,research=False,world_simulations=False))
                agent.tick()
                self.assertEqual(brain.world.observations,0)
            finally:brain.close()


if __name__=='__main__':unittest.main()
