"""Public-source research and safe teacher-validated learning regression tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nord_brain.internet import _DDG, _to_results, _canonical_url, _safe_query, public_question
from nord_brain.conversation import interpret
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore, TransformerConfig
from nord_brain.natural_math import read_math_question

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)
SOURCES=[{
    'title':'Elon Musk | Britannica',
    'url':'https://www.britannica.com/money/Elon-Musk',
    'summary':'Elon Musk é um empresário conhecido pela Tesla e pela SpaceX, entre outros projetos.',
    'source_tier':'trusted_discovery',
    'source_label':'britannica.com',
    'provider':'duckduckgo-html',
}]

class SearchTests(unittest.TestCase):
    def test_public_questions_and_math_followups(self):
        self.assertEqual(public_question('quem e o Elon Musk'),'Elon Musk')
        self.assertEqual(public_question('Quem é Elon Musk?'),'Elon Musk')
        self.assertIsNone(public_question('o que sabes'))
        self.assertEqual(interpret('quem é Elon Musk').kind,'question')
        self.assertIn('8',read_math_question('e 3+5')[1].answer)
        self.assertIn('12',read_math_question('e 10 + 2')[1].answer)

    def test_html_results_parser_and_source_screening(self):
        html='''<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.britannica.com%2Fmoney%2FElon-Musk">Elon Musk - Britannica</a>
        <a class="result__snippet">Elon Musk is an entrepreneur connected to Tesla and SpaceX.</a></div>
        <div><a class="result__a" href="https://en.wikipedia.org/wiki/Elon_Musk">Wiki</a>
        <div class="result__snippet">Wikipedia page about the subject which is excluded.</div></div>'''
        parser=_DDG();parser.feed(html)
        results=_to_results(parser.rows,'Elon Musk','duckduckgo-html')
        self.assertEqual(len(results),1)
        self.assertEqual(results[0]['source_tier'],'trusted_discovery')
        self.assertIn('britannica.com',results[0]['url'])
        self.assertIsNone(_canonical_url('http://127.0.0.1/private'))
        self.assertIsNone(_canonical_url('https://en.wikipedia.org/wiki/Elon_Musk'))
        self.assertIsNone(_safe_query('my api key=abcdef'))
        self.assertIsNone(_safe_query('ricardo@example.com'))

class RuntimeTests(unittest.TestCase):
    def brain(self,path):
        b=Brain(Path(path),start_autonomy=False)
        b.core=NeuralCore(Path(path)/'brain-v2.pt', CONFIG)
        return b

    def test_research_does_not_fabricate_or_train_until_confirmed(self):
        with tempfile.TemporaryDirectory() as path:
            b=self.brain(path)
            try:
                before=b.core.steps
                with patch('nord_brain.runtime.search',return_value=SOURCES):
                    msg,source=b.chat('quem é Elon Musk')
                self.assertIn('Britannica',msg)
                self.assertIn('Fonte:',msg)
                self.assertIn('não confirmadas',msg)
                self.assertIn('pesquisa pública',source)
                self.assertEqual(b.core.steps,before)
                message,origin=b.chat('confirma fonte 1')
                self.assertIn('Confirmação registada',message)
                self.assertIn('treino neuronal',origin)
                self.assertGreater(b.core.steps,before)
                self.assertIn('empresário',b.chat('quem é Elon Musk')[0])
            finally:b.close()

    def test_missing_web_results_refuses_to_hallucinate(self):
        with tempfile.TemporaryDirectory() as path:
            b=self.brain(path)
            try:
                with patch('nord_brain.runtime.search',return_value=[]):
                    reply,origin=b.chat('quem é Elon Musk')
                self.assertIn('Não encontrei fontes',reply)
                self.assertNotIn('nord> 3',reply)
                self.assertIn('pesquisa sem resultados',origin)
                msg,_=b.chat('confirma fonte 1')
                self.assertIn('Não tenho uma pesquisa',msg)
            finally:b.close()

    def test_search_explicit_and_concept_training_supports_cognitive_lab(self):
        with tempfile.TemporaryDirectory() as path:
            b=self.brain(path)
            try:
                with patch('nord_brain.runtime.search',return_value=SOURCES):
                    result,_=b.chat('pesquisa sobre Elon Musk')
                self.assertIn('Britannica',result)
                b.chat('confirma fonte 1')
                self.assertIsNotNone(b.memory.lookup_concept('Elon Musk'))
                # Confirmed web research is sourced, not equivalent to an
                # independently taught teacher concept in the reasoning lab.
                self.assertIsNone(b.cognition.candidate('linguagem'))
                self.assertEqual(b.memory.lookup_concept('Elon Musk'), SOURCES[0]['summary'])
            finally:b.close()

    def test_math_is_not_a_conversation_error(self):
        with tempfile.TemporaryDirectory() as path:
            b=self.brain(path)
            try:
                self.assertIn('8',b.chat('e 3+5')[0])
                self.assertIn('12',b.chat('e 10 + 2')[0])
                self.assertIn('capacidade',b.chat('o que sabes')[0].lower())
            finally:b.close()

if __name__=='__main__':unittest.main()
