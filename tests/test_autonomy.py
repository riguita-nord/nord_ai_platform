import tempfile
import threading
import unittest
from pathlib import Path
import torch

from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore, TransformerConfig
from nord_brain.autonomy import Autonomy, AutoSettings, growth_candidate
from nord_brain.conversation import interpret


TEST_CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)


class NaturalTeachingTests(unittest.TestCase):
    def prepare(self, folder):
        brain=Brain(Path(folder),start_autonomy=False)
        brain.core=NeuralCore(Path(folder)/'brain-v2.pt', TEST_CONFIG)
        return brain

    def test_uncommanded_teaching(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=self.prepare(tmp)
            start=brain.core.steps
            say,origin=brain.chat('O Python é uma linguagem de programação.')
            self.assertIn('Aprendi',say)
            self.assertGreater(brain.core.steps,start)
            self.assertIn('linguagem de programação',brain.chat('O que é Python?')[0])
            self.assertIn('conceito',brain.chat('O que é Python?')[1])
            brain.close()
            again=Brain(Path(tmp),start_autonomy=False)
            self.assertIn('linguagem de programação',again.chat('O que é Python?')[0])
            again.close()

    def test_plain_english_like_correction_in_portuguese(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=self.prepare(tmp)
            response,_=brain.chat('Quem inventou o motor X?')
            self.assertIn('não sei',response.lower())
            response,_=brain.chat('Não, a resposta correta é Maria.')
            self.assertIn('correção',response)
            self.assertEqual(brain.memory.exact('Quem inventou o motor X?'),'Maria.')
            brain.close()

    def test_name_and_direct_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=self.prepare(tmp)
            brain.chat('Chamo-me Ricardo.')
            self.assertIn('Ricardo',brain.chat('Como me chamo?')[0])
            brain.chat('Quando eu perguntar qual é a tua cor?, responde azul.')
            self.assertIn('azul',brain.chat('qual é a tua cor?')[0])
            brain.close()

    def test_long_teaching_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            core=NeuralCore(Path(tmp)/'core.pt', TEST_CONFIG)
            long_answer='Pequenos exemplos de aprendizagem. ' * 10
            windows=core._training_windows('O que significa isto?',long_answer,TEST_CONFIG.context)
            self.assertGreater(len(windows),1)
            for ids,labels in windows:
                self.assertLessEqual(ids.size(1),TEST_CONFIG.context)
                self.assertGreater(int((labels!=-100).sum()),0)
            loss=core.train('O que significa isto?',long_answer,epochs=1)
            self.assertGreater(loss,0)
            self.assertGreater(core.score([('O que significa isto?',long_answer)]),0)

    def test_parser_is_honest_about_unknowns(self):
        self.assertEqual(interpret('O Python é uma linguagem de programação.').kind,'fact')
        self.assertEqual(interpret('Como estás?').kind,'question')
        self.assertEqual(interpret('Não, a resposta correta é 4','Quanto é 2 + 2?').answer,'4')


class AutonomousTests(unittest.TestCase):
    def test_growth_policy_is_conservative(self):
        args=dict(steps=50,lesson_count=50,holdout_count=10,params=200000,width=192,max_params=2000000,max_width=1024)
        self.assertTrue(growth_candidate([5.0,4.99,4.98],**args))
        self.assertFalse(growth_candidate([5,4,3],**args))
        self.assertFalse(growth_candidate([5,5,5],**{**args,'holdout_count':2}))
        self.assertFalse(growth_candidate([5,5,5],**{**args,'params':2100000}))

    def test_autonomy_tick_trains_without_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=Brain(Path(tmp),start_autonomy=False)
            brain.core=NeuralCore(Path(tmp)/'brain-v2.pt',TEST_CONFIG)
            before=brain.core.steps
            agent=Autonomy(brain,AutoSettings(training=True,experiments=True,growth=False,research=False))
            t=threading.Thread(target=agent.tick)
            t.start();t.join(timeout=15)
            self.assertFalse(t.is_alive())
            self.assertGreater(brain.core.steps,before)
            self.assertGreater(brain.memory.lesson_count(),0)
            self.assertGreaterEqual(len(brain.memory.tests()),6)
            self.assertNotEqual(brain.memory.get_meta('curriculum_cursor'),None)
            brain.close()

    def test_growth_trial_compares_against_unexpanded_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=Brain(Path(tmp),start_autonomy=False)
            brain.core=NeuralCore(Path(tmp)/'brain-v2.pt',TEST_CONFIG)
            for i in range(14):
                brain.teach(f'Objeto {i}',f'Item {i}',epochs=1)
            for i in range(10):
                brain.memory.holdout(f'Número de teste {i}?',str(i))
            brain.core.steps=45
            for score in (5.0,5.0,5.0):
                brain.memory.event('eval_loss','holdouts',score)
            agent=Autonomy(brain,AutoSettings(training=True,growth=True,
                    experiments=False,research=False,max_params=2000000,max_width=1024))
            old=brain.core.parameter_count
            agent._maybe_grow(brain.memory.tests(10))
            events=brain.memory.recent_events(3)
            self.assertTrue(any(kind.startswith('growth_') for kind,_,_ in events))
            self.assertTrue(brain.core.parameter_count==old or brain.core.parameter_count>old)
            self.assertIn(brain.core.hidden_size,(64,80))
            brain.close()

    def test_no_internet_trains_without_external_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain=Brain(Path(tmp),start_autonomy=False)
            brain.core=NeuralCore(Path(tmp)/'brain-v2.pt',TEST_CONFIG)
            brain.memory.unknown('teoria do céu')
            agent=Autonomy(brain,AutoSettings(training=False,research=False))
            agent.tick()
            self.assertEqual(brain.core.steps,0)
            self.assertIsNone(brain.memory.observation_for_topic('teoria do céu'))
            brain.close()

if __name__=='__main__':unittest.main()
