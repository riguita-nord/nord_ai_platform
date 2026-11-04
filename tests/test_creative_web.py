"""Offline regression tests for creative routing, local model and public research."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nord_brain.creative import creative_request,story
from nord_brain.language_backend import LocalLanguage,_endpoint
from nord_brain.internet import _bing_rss_search,search_detailed,search_diagnostic
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore,TransformerConfig
from nord_brain.storage import Memory

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)

class FakeResponse:
    status=200
    def __init__(self,data):self.data=data
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def read(self,n=600001):return self.data[:n]

class LocalLanguageTests(unittest.TestCase):
    def test_endpoint_cannot_exfiltrate_to_remote_server(self):
        with patch.dict('os.environ',{'NORD_LLM_URL':'https://evil.example/v1/chat/completions'}):
            with self.assertRaises(ValueError):_endpoint()
        with patch.dict('os.environ',{'NORD_LLM_URL':'http://127.0.0.1:8081/v1/chat/completions'}):
            self.assertIn('127.0.0.1',_endpoint())

    def test_local_pretrained_model_is_not_mislabeled_as_nord_training(self):
        payload=json.dumps({'choices':[{'message':{'content':'Uma história original, com princípio e fim.'}}]}).encode()
        instance=LocalLanguage()
        class Opener:
            def open(self,req,timeout):
                self.payload=json.loads(req.data)
                self.url=req.full_url
                return FakeResponse(payload)
        opener=Opener()
        with patch('nord_brain.language_backend.request.build_opener',return_value=opener):
            output=instance.generate('Conta-me uma história',creative=True)
        self.assertIn('história original',output)
        self.assertEqual(opener.payload['messages'][-1]['role'],'user')
        self.assertEqual(opener.payload['stream'],False)
        self.assertEqual(len(instance.history),2)
        self.assertTrue(opener.url.startswith('http://127.0.0.1:'))

class CreativeTests(unittest.TestCase):
    def make_brain(self,path):
        b=Brain(Path(path),start_autonomy=False)
        b.core=NeuralCore(Path(path)/'brain-v2.pt',CONFIG)
        return b

    def test_story_intent_and_offline_composition(self):
        for q in ('conta-me uma historia','conta-me uma hitoria','Inventa um conto sobre robôs'):
            self.assertIsNotNone(creative_request(q),q)
        self.assertIsNone(creative_request('Qual é a história de Portugal?'))
        self.assertIn('Por fim',story('um projeto'))
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.make_brain(d)
            try:
                before=b.core.steps
                text,source=b.chat('conta-me uma hitoria')
                self.assertGreater(len(text),300)
                self.assertIn('narrativa',source)
                self.assertEqual(b.core.steps,before)
                self.assertFalse(b.awaiting_explanation)
                self.assertIsNone(b.memory.exact('conta-me uma hitoria'))
            finally:b.close()

    def test_repeated_greetings_change_without_fabricating_neural_fluency(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.make_brain(d)
            try:
                replies=[b.chat('ola')[0] for _ in range(6)]
                self.assertEqual(len(set(replies[:5])),5)
                self.assertEqual(replies[0],replies[5])
            finally:b.close()

    def test_local_language_responds_but_math_stays_verified(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.make_brain(d)
            try:
                with patch.object(b.language,'generate',return_value='Era uma vez um dragão.'):
                    response,source=b.chat('fala comigo sobre aventuras')
                    self.assertIn('dragão',response)
                    self.assertIn('local',source)
                    math,_=b.chat('2+2')
                    self.assertIn('4',math)
            finally:b.close()

class SearchFallbackTests(unittest.TestCase):
    def test_rss_results_have_source_and_screening(self):
        xml=b'<rss><channel><item><title>Elon Musk | Britannica</title><description>Elon Musk is an entrepreneur associated with Tesla and SpaceX.</description><link>https://www.britannica.com/money/Elon-Musk</link></item></channel></rss>'
        with patch('nord_brain.internet._read_response',return_value=xml):
            rows=_bing_rss_search('Elon Musk')
        self.assertEqual(len(rows),1)
        self.assertIn('britannica',rows[0][2])

    def test_bing_is_used_when_duckduckgo_empty(self):
        rows=[('Elon Musk | Britannica','Elon Musk is an entrepreneur associated with Tesla and SpaceX.','https://www.britannica.com/money/Elon-Musk')]
        with patch('nord_brain.internet._ddg_search',return_value=[]),patch('nord_brain.internet._bing_rss_search',return_value=rows):
            results,status=search_detailed('Elon Musk')
            self.assertEqual(len(results),1)
            self.assertTrue(any('Bing RSS' in msg for msg in status))
        with patch('nord_brain.internet._ddg_search',side_effect=OSError('block')), \
             patch('nord_brain.internet._bing_rss_search',return_value=[]):
            self.assertIn('DuckDuckGo HTML: OSError',search_diagnostic('Elon Musk'))

class TeacherPriorityTests(unittest.TestCase):
    def test_teacher_memory_wins_over_recent_external_data(self):
        with tempfile.TemporaryDirectory() as d:
            mem=Memory(Path(d)/'memory.sqlite3')
            try:
                mem.teach('O que é a cor da luz?','o que eu ensinei','teacher')
                mem.teach('O que é a cor da luz?','outro resultado web','official_primary_excerpt')
                self.assertEqual(mem.exact('O que é a cor da luz?'),'o que eu ensinei')
                mem.concept('objeto','explicação humana',origin='teacher')
                mem.concept('objeto','outra explicação web',origin='teacher_confirmed_research')
                self.assertEqual(mem.lookup_concept('objeto'),'explicação humana')
                self.assertNotIn(('O que é a cor da luz?','outro resultado web'),mem.lessons())
            finally:mem.close()

if __name__=='__main__':unittest.main()
