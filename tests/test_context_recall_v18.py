"""Regression tests for conversation-wide, query-dependent context selection."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nord_brain.dialogue import DialogueMemory, followup_kind
from nord_brain.language_backend import LocalLanguage


class ConversationRecallTests(unittest.TestCase):
    def test_natural_followups_do_not_require_exact_training_phrases(self):
        for text in ('faz mais extenso', 'consegues fazer maior',
                     'podes fazer uma história maior', 'faz mais detalhado',
                     'continua daí', 'consegues explicar melhor'):
            self.assertIsNotNone(followup_kind(text), text)
        for text in ('2+2', 'e 3+5', 'o que sabes fazer', 'uma história é ficção'):
            self.assertIsNone(followup_kind(text), text)

    def test_searches_older_conversations_not_only_last_five(self):
        with tempfile.TemporaryDirectory() as temp:
            memory=DialogueMemory(Path(temp))
            memory.add('Quero construir um robô elétrico',
                       'Precisamos de bateria, rodas e sensores.',
                       'dialogue','dialogue','robô elétrico')
            for idx in range(14):
                memory.add('Quanto é '+str(idx)+' mais dois?',
                           'Exemplo matemático '+str(idx),
                           'math','math',str(idx))
            messages=memory.context_messages('como melhorar o robô elétrico?')
            self.assertTrue(any('construir um robô' in m['content'] for m in messages))
            self.assertTrue(any('bateria' in m['content'] for m in messages))
            self.assertIn('mais dois?', messages[-2]['content'])
            memory.close()

    def test_preserves_current_subject_when_request_uses_pronouns(self):
        with tempfile.TemporaryDirectory() as temp:
            memory=DialogueMemory(Path(temp))
            memory.add('Tenho um projeto de drones', 'Falei de drones autónomos.',
                       'dialogue','dialogue','projeto de drones')
            for idx in range(10):
                memory.add('Uma pergunta de matemática '+str(idx),
                           'Resposta '+str(idx),'math','math','matemática')
            memory.add('E como melhorar isso?', 'Podemos testar um protótipo.',
                       'dialogue','dialogue','projeto de drones')
            messages=memory.context_messages('e as desvantagens disso?')
            self.assertTrue(any('projeto de drones' in m['content'] for m in messages))
            memory.close()

    def test_story_recall_prefers_full_narrative_not_last_short_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            memory=DialogueMemory(Path(temp))
            body='A história de Nuno.\n\n'+'Nuno descobriu uma máquina misteriosa. '*15
            memory.add('Conta-me uma história', body,
                       'composição narrativa procedural','story')
            memory.add('Quem era ele?', 'Era Nuno.',
                       'continuidade contextual da história','story','Conta-me uma história')
            self.assertEqual(memory.latest_story()['reply'],body)
            memory.close()

    def test_history_is_durable_but_bounded_for_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            memory=DialogueMemory(Path(temp))
            for idx in range(490):
                memory.add('Pergunta '+str(idx),'Resposta','dialogue','dialogue')
            self.assertEqual(memory.db.execute('SELECT COUNT(*) FROM turns').fetchone()[0],480)
            memory.close()
            reopened=DialogueMemory(Path(temp))
            self.assertEqual(reopened.latest()['question'],'Pergunta 489')
            reopened.close()

    def test_llm_requests_query_specific_context(self):
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self,limit):
                return json.dumps({'choices':[{'message':{'content':'Percebi o contexto.'}}]}).encode()
        class Opener:
            def open(self,req,timeout):return Response()
        seen=[]
        language=LocalLanguage()
        def provider(prompt):
            seen.append(prompt)
            return [{'role':'user','content':'Tenho um projeto de robôs.'},
                    {'role':'assistant','content':'Qual é o objetivo?'}]
        language.context_provider=provider
        with patch.dict('os.environ',{'NORD_LLM':'on'}),patch(
                'nord_brain.language_backend.request.build_opener',return_value=Opener()):
            answer=language.generate('e o que podemos melhorar?')
        self.assertEqual(answer,'Percebi o contexto.')
        self.assertEqual(seen,['e o que podemos melhorar?'])


if __name__=='__main__':
    unittest.main()
