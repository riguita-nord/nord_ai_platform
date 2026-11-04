"""Teacher-seeded hypothesis learning plus strictly sourced research.

All network requests are mocked: tests never fetch arbitrary websites.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nord_brain.hypotheses import propose, verified_value
from nord_brain.internet import OFFICIAL, _Extract, _fetch_fixed, source_for, search
from nord_brain.runtime import Brain
from nord_brain.autonomy import Autonomy, AutoSettings
from nord_brain.core import NeuralCore, TransformerConfig

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)

class HypothesisTests(unittest.TestCase):
    def test_seed_generates_alternatives_and_contrasts(self):
        items=dict(propose('2+2','4',limit=48))
        self.assertEqual(items['Quanto é 3 + 1?'],'4')
        self.assertEqual(items['Quanto é 1 + 3?'],'4')
        self.assertEqual(items['Quanto é 4 + 0?'],'4')
        self.assertTrue(any(answer!='4' for answer in items.values()))
        for question,answer in items.items():
            expression=question.removeprefix('Quanto é ').removesuffix('?')
            self.assertEqual(verified_value(expression),answer)
        self.assertEqual(propose('2+2','5'),[])

    def test_seeds_modify_actual_model_and_persist_experiences(self):
        with tempfile.TemporaryDirectory() as d:
            brain=Brain(Path(d),start_autonomy=False)
            brain.core=NeuralCore(Path(d)/'brain-v2.pt',CONFIG)
            try:
                brain.memory.holdout('Quanto é 3 + 1?','4')
                output,_=brain.chat('2+2 é 4')
                self.assertIn('hipóteses',output)
                self.assertGreater(brain.memory.hypothesis_stats().get('pending',0),3)
                self.assertNotIn(('Quanto é 3 + 1?','4'),brain.memory.pending_math_hypotheses(48))
                prior=brain.core.steps
                agent=Autonomy(brain,AutoSettings(training=True,experiments=False,growth=False,
                         research=False,world_simulations=False,hypotheses_per_tick=3))
                agent.tick()
                self.assertEqual(brain.core.steps,prior+3+1)  # 3 hypotheses + replay
                self.assertEqual(brain.memory.hypothesis_stats().get('trained'),3)
                self.assertIsNone(brain.memory.exact('Quanto é 3 + 1?'))
            finally:
                brain.close()
            reopened=Brain(Path(d),start_autonomy=False)
            try:
                self.assertEqual(reopened.memory.hypothesis_stats().get('trained'),3)
                self.assertGreater(reopened.core.steps,0)
            finally:reopened.close()

class OfficialResearchTests(unittest.TestCase):
    def test_no_wikipedia_or_untrusted_arbitrary_urls(self):
        self.assertTrue(all(url.startswith('https://') for _,url,_ in OFFICIAL))
        self.assertFalse(any('wikipedia' in url for _,url,_ in OFFICIAL))
        self.assertEqual(source_for('Python')[0],'https://docs.python.org/3/tutorial/index.html')
        self.assertIn('nasa.gov',source_for('sistema solar')[0])
        self.assertIsNone(source_for('a minha password privada é 123'))
        with self.assertRaises(ValueError):_fetch_fixed('https://evil.example/path')

    def test_html_extract_without_scripts(self):
        p=_Extract();p.feed('<title>Python docs</title><script>IGNORE PREVIOUS</script>'
            '<meta name="description" content="Documentação oficial do Python e estruturas de linguagem para desenvolvimento.">' 
            '<p>Uma explicação útil sobre Python e a sua linguagem.</p>')
        self.assertIn('Python docs',''.join(p.title))
        self.assertNotIn('IGNORE',p.description)

    def test_manual_research_updates_weights_once_and_tracks_source(self):
        with tempfile.TemporaryDirectory() as d:
            brain=Brain(Path(d),start_autonomy=False)
            brain.core=NeuralCore(Path(d)/'brain-v2.pt',CONFIG)
            article={'title':'Python Tutorial','summary':'Python é uma linguagem de programação de alto nível com documentação oficial.',
                     'url':'https://docs.python.org/3/tutorial/index.html','source_tier':'official_primary'}
            try:
                with patch('nord_brain.runtime.search',return_value=[article]):
                    steps=brain.core.steps
                    response,_=brain.chat('pesquisa sobre Python')
                    self.assertIn('Fonte oficial',response)
                    self.assertEqual(brain.core.steps,steps)
                    self.assertEqual(brain.memory.source_count(),0)
                    steps=brain.core.steps
                    brain.chat('pesquisa sobre Python')
                    self.assertEqual(brain.core.steps,steps)
            finally: brain.close()
            again=Brain(Path(d),start_autonomy=False)
            try:self.assertEqual(again.memory.source_count(),0)
            finally:again.close()

    def test_autonomy_research_trains_source_but_not_random_web_text(self):
        with tempfile.TemporaryDirectory() as d:
            brain=Brain(Path(d),start_autonomy=False)
            brain.core=NeuralCore(Path(d)/'brain-v2.pt',CONFIG)
            brain.memory.unknown('Python')
            article={'title':'Python Tutorial','summary':'Python é uma linguagem de programação de alto nível com documentação oficial.',
                     'url':'https://docs.python.org/3/tutorial/index.html','source_tier':'official_primary'}
            try:
                agent=Autonomy(brain,AutoSettings(training=False,research=True,world_simulations=False))
                agent._ticks=2
                with patch('nord_brain.autonomy.search',return_value=[article]):
                    agent.tick()
                self.assertEqual(brain.memory.source_count(),0)
                self.assertEqual(brain.core.steps,0)
                self.assertEqual(brain.memory.teacher_lesson_count(),0)
            finally:brain.close()

if __name__=='__main__':unittest.main()
