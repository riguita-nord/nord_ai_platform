"""Teacher-directed counterfactual simulation with persistent review controls."""
from pathlib import Path
import tempfile
import unittest

from nord_brain.autonomy import Autonomy, AutoSettings
from nord_brain.core import NeuralCore, TransformerConfig
from nord_brain.hypotheses import explore_theories, verified_value
from nord_brain.runtime import Brain

CONFIG = TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)


class TheoryGenerationTests(unittest.TestCase):
    def test_plus_and_minus_one_lead_to_new_results(self):
        cases = {q:(ans,description) for q,ans,description in explore_theories('2+2','4')}
        self.assertEqual(cases['Quanto é 3 + 2?'][0],'5')
        self.assertEqual(cases['Quanto é 2 + 3?'][0],'5')
        self.assertEqual(cases['Quanto é 1 + 2?'][0],'3')
        self.assertEqual(cases['Quanto é 3 + 1?'][0],'4')
        self.assertIn('acrescentar',cases['Quanto é 3 + 2?'][1])
        self.assertGreaterEqual(len({v[0] for v in cases.values()}),4)
        for q,(a,_) in cases.items():
            self.assertEqual(verified_value(q.removeprefix('Quanto é ').removesuffix('?')),a)

    def test_fixed_neural_holdout_never_enters_generated_training(self):
        from nord_brain.experiments import reserved_math_probe
        self.assertTrue(reserved_math_probe('Quanto é 4 + 4?'))
        with tempfile.TemporaryDirectory() as tmp:
            b=Brain(Path(tmp),start_autonomy=False)
            b.core=NeuralCore(Path(tmp)/'brain-v2.pt',CONFIG)
            try:
                b.chat('2+2 é 4')
                self.assertNotIn('Quanto é 4 + 4?',
                    [row[0] for row in b.memory.theory_details(limit=40)])
                self.assertIsNone(b.memory.exact('Quanto é 4 + 4?'))
            finally:b.close()

    def test_wrong_seed_is_not_used_as_theory(self):
        self.assertEqual(explore_theories('2+2','9'),[])
        self.assertEqual(explore_theories('foo','4'),[])


class BrainTheoryTests(unittest.TestCase):
    def make_brain(self, directory):
        b=Brain(Path(directory),start_autonomy=False)
        b.core=NeuralCore(Path(directory)/'brain-v2.pt',CONFIG)
        return b

    def agent(self,b):
        return Autonomy(b,AutoSettings(training=True,experiments=False,
               world_simulations=False,cognitive_lab=False,growth=False,research=False,
               hypotheses_per_tick=8))

    def test_teacher_can_require_approval_and_select_one_theory(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.make_brain(tmp)
            try:
                response,_=b.chat('quero aprovar as hipóteses')
                self.assertIn('Modo de revisão',response)
                b.chat('2+2 é 4')
                theories=b.memory.theory_details(limit=40)
                self.assertTrue(any(q=='Quanto é 3 + 2?' and state=='review' and 'acrescentar' in why
                                    for q,_,state,why,_ in theories))
                before=b.core.steps
                self.agent(b).tick()
                self.assertGreaterEqual(b.core.steps,before) # manual example may replay
                self.assertEqual(b.memory.hypothesis_stats().get('trained',0),0)
                approval,_=b.chat('aprova hipótese 3+2')
                self.assertIn('Aprovada',approval)
                # only selected theory is approved; another theory stays in review.
                self.assertEqual(b.memory.pending_math_hypotheses(9),[('Quanto é 3 + 2?','5')])
                self.agent(b).tick()
                self.assertEqual(b.memory.hypothesis_stats().get('trained',0),1)
                self.assertTrue(b.memory.exact('Quanto é 3 + 2?')=='5')
                self.assertGreater(b.core.steps,before)
            finally:
                b.close()

    def test_rejection_blocks_replay_and_future_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.make_brain(tmp)
            try:
                b.chat('2+2 é 4')
                refusal,_=b.chat('rejeita hipótese 3+2')
                self.assertIn('Bloqueei',refusal)
                self.assertTrue(b.memory.is_training_blocked('Quanto é 3 + 2?'))
                self.assertNotIn('Quanto é 3 + 2?', [q for q,_ in b.memory.pending_math_hypotheses(12)])
                steps=b.core.steps
                self.agent(b).tick()
                self.assertTrue(b.memory.exact('Quanto é 3 + 2?') is None)
                self.assertGreaterEqual(b.core.steps,steps)
                with self.assertRaises(ValueError):
                    b.teach('Quanto é 3 + 2?','5')
            finally:
                b.close()
            reopened=Brain(Path(tmp),start_autonomy=False)
            try:
                self.assertTrue(reopened.memory.is_training_blocked('Quanto é 3 + 2?'))
            finally:reopened.close()

    def test_pause_reliably_disables_autonomous_learning_and_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.make_brain(tmp)
            try:
                b.chat('2+2 é 4')
                result,_=b.chat('pausa aprendizagem')
                self.assertIn('Parei',result)
                before=(b.core.steps,b.world.train_steps,b.memory.hypothesis_stats().get('trained',0))
                agent=Autonomy(b,AutoSettings(training=True,experiments=True,
                    world_simulations=True,cognitive_lab=True,research=False,growth=False))
                agent.tick()
                self.assertEqual((b.core.steps,b.world.train_steps,b.memory.hypothesis_stats().get('trained',0)),before)
                b.chat('retoma aprendizagem')
                agent.tick()
                self.assertGreater(b.core.steps,before[0])
            finally:b.close()

    def test_wrong_math_correction_rejected_and_known_good_one_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.make_brain(tmp)
            try:
                response,_=b.chat('3+2 é 8')
                self.assertIn('não confirma',response)
                self.assertEqual(b.memory.hypothesis_stats().get('pending',0),0)
                response,_=b.chat('3+2 é 5')
                self.assertIn('Correto',response)
                self.assertGreater(b.memory.hypothesis_stats().get('pending',0),2)
            finally:b.close()

    def test_review_mode_persistent_across_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.make_brain(tmp)
            try:
                b.chat('quero aprovar as hipóteses')
                b.chat('2+2 é 4')
            finally:b.close()
            b2=Brain(Path(tmp),start_autonomy=False)
            try:
                self.assertTrue(b2.memory.review_required())
                self.assertTrue(b2.memory.hypothesis_stats().get('review',0)>0)
                view,_=b2.chat('mostra teorias')
                self.assertIn('3 + 2',view)
                self.assertIn('acrescentar',view)
            finally:b2.close()

if __name__=='__main__':unittest.main()
