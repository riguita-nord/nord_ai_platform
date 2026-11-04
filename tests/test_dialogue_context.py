"""Regression tests for anaphoric conversational follow-ups and story revisions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nord_brain.dialogue import DialogueMemory,followup_kind
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore,TransformerConfig

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)


class DialogueTests(unittest.TestCase):
    def brain(self,root):
        b=Brain(Path(root),start_autonomy=False)
        b.core=NeuralCore(Path(root)/'brain-v2.pt',CONFIG)
        return b

    def test_exact_transcript_rewrites_the_same_story_not_a_new_topic(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.brain(root)
            try:
                initial_steps=b.core.steps
                first,origin=b.chat('conta-me uma historia')
                self.assertIn('narrativa',origin)
                self.assertGreater(len(first),250)
                longer,origin=b.chat('faz uma mais longa')
                self.assertGreater(len(longer),len(first)*1.6)
                self.assertIn('ficção contextual',origin)
                name=first.split('\n\n',1)[1].split(',',1)[0]
                self.assertIn(name,longer)
                self.assertIn('encontrou',longer)
                self.assertEqual(b.core.steps,initial_steps)
                self.assertFalse(b.awaiting_explanation)
                self.assertIsNone(b.memory.exact('faz uma mais longa'))
                self.assertEqual(b.dialogue.latest()['focus'],'story')
            finally:b.close()

    def test_user_transcript_extenso_and_consegues_maior_use_story_context(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.brain(root)
            try:
                steps=b.core.steps
                first,_=b.chat('conta-me uma historia')
                name=first.split('\n\n',1)[1].split(',',1)[0]
                second,_=b.chat('faz mais extenso')
                third,_=b.chat('consegues fazer maior')
                self.assertGreater(len(second),len(first))
                self.assertGreater(len(third),len(second))
                self.assertIn(name,second)
                self.assertIn(name,third)
                self.assertFalse(b.awaiting_explanation)
                self.assertIsNone(b.last_question)
                self.assertIsNone(b.memory.exact('consegues fazer maior'))
                self.assertEqual(b.core.steps,steps)
            finally:b.close()

    def test_character_question_then_story_expansion_uses_full_story(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.brain(root)
            try:
                first,_=b.chat('conta-me uma historia')
                who,_=b.chat('qual era o nome da personagem?')
                self.assertLess(len(who),len(first))
                expanded,_=b.chat('faz mais extenso')
                self.assertGreater(len(expanded),len(first))
                self.assertIn(first.split('\n\n',1)[1].split(',',1)[0],expanded)
            finally:b.close()

    def test_continuation_command_is_not_an_automatic_teacher_label(self):
        from nord_brain.conversation import interpret
        for message in ('faz mais extenso','consegues fazer maior',
                        'e o que achas disso?','consegues explicar melhor'):
            intent=interpret(message,'Pergunta antiga?',True)
            self.assertNotIn(intent.kind,('explanation','correction','teach','fact'),message)

    def test_live_transcript_story_to_chapters_and_followup(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.brain(root)
            try:
                training_steps=b.core.steps
                first,_=b.chat('conta-me uma historia')
                longer,_=b.chat('faz mais extenso')
                chaptered,origin=b.chat('com capitulos e etc')
                self.assertGreater(len(longer),len(first))
                self.assertIn('Capítulo 1',chaptered)
                self.assertIn('Capítulo 2',chaptered)
                self.assertIn('Capítulo 3',chaptered)
                self.assertEqual(b.stories.active()['written'],3)
                self.assertIn(first.split('\n\n',1)[1].split(',',1)[0],chaptered)
                self.assertEqual(b.dialogue.latest()['focus'],'book')
                self.assertFalse(b.awaiting_explanation)
                self.assertIsNone(b.memory.exact('com capitulos e etc'))
                self.assertEqual(b.core.steps,training_steps)
                next_chapter,_=b.chat('continua a historia')
                self.assertIn('Capítulo 4',next_chapter)
                self.assertEqual(b.stories.active()['written'],4)
            finally:b.close()

    def test_context_refinement_in_story_is_not_teacher_training(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.brain(root)
            try:
                original,_=b.chat('conta-me uma historia')
                count=b.memory.lesson_count()
                revised,origin=b.chat('com mais diálogos')
                self.assertGreater(len(revised),len(original))
                self.assertIn('Nuno' if 'Nuno' in original else original.split('\n\n',1)[1].split(',',1)[0],revised)
                self.assertEqual(b.memory.lesson_count(),count)
                self.assertFalse(b.awaiting_explanation)
            finally:b.close()

    def test_continue_rewrite_and_other_story_followups(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.brain(root)
            try:
                first,_=b.chat('conta-me uma historia')
                name=first.split('\n\n',1)[1].split(',',1)[0]
                continuation,_=b.chat('continua a história')
                self.assertIn(name,continuation)
                self.assertIn('Um novo capítulo',continuation)
                ending,_=b.chat('muda o final')
                self.assertIn(name,ending)
                self.assertGreater(len(ending),100)
            finally:b.close()

    def test_dialogue_survives_restart_and_never_teaches_itself(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.brain(root)
            story,_=b.chat('conta-me uma historia')
            name=story.split('\n\n',1)[1].split(',',1)[0]
            b.close()
            b=self.brain(root)
            try:
                revised,_=b.chat('faz uma mais longa')
                self.assertIn(name,revised)
                self.assertGreater(len(revised),len(story))
                self.assertEqual(b.memory.lesson_count(),0)
                self.assertEqual(b.dialogue.latest()['focus'],'story')
            finally:b.close()

    def test_new_topic_prevents_unrelated_story_rewrite(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.brain(root)
            try:
                b.chat('conta-me uma historia')
                answer,_=b.chat('4+4')
                self.assertIn('8',answer)
                follow,_=b.chat('faz uma mais longa')
                self.assertNotIn('ficção contextual',b.dialogue.latest()['origin'])
                self.assertNotIn('Uma porta',follow)
            finally:b.close()

    def test_local_language_receives_actual_prior_story(self):
        with tempfile.TemporaryDirectory() as root:
            b=self.brain(root)
            try:
                with patch.object(b.language,'generate',side_effect=[None,'Uma versão nova e muito longa. '*100]) as generate:
                    story,_=b.chat('conta-me uma historia')
                    expanded,_=b.chat('faz uma mais longa')
                    self.assertIn('Uma versão nova',expanded)
                    second=generate.call_args_list[1]
                    self.assertIn(story[:80],second.args[0])
                    self.assertTrue(second.kwargs['creative'])
                    self.assertFalse(second.kwargs['remember'])
            finally:b.close()

    def test_book_chapter_expansion_edits_current_chapter_not_new_book(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.brain(root)
            try:
                b.chat('cria um livro sobre um farol misterioso')
                first=b.stories.chapter(1)[1]
                response,origin=b.chat('faz uma mais longa')
                updated=b.stories.chapter(1)[1]
                self.assertIn('capítulo revisto',origin)
                self.assertGreater(len(updated),len(first))
                self.assertEqual(b.stories.active()['written'],1)
                self.assertEqual(len(b.stories.list_books()),1)
                self.assertIn(updated[:30],response)
                next_chapter,_=b.chat('continua a história')
                self.assertIn('Capítulo 2',next_chapter)
            finally:b.close()

    def test_generic_explanation_uses_last_topic_and_not_training(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.brain(root)
            try:
                b.chat('4+4')
                reply,origin=b.chat('explica melhor')
                self.assertIn('4+4',reply)
                self.assertIn('continuidade contextual',origin)
                self.assertIsNone(b.memory.exact('explica melhor'))
            finally:b.close()

    def test_followup_parser_does_not_swallow_unrelated_text(self):
        self.assertEqual(followup_kind('faz uma mais longa'),'longer')
        self.assertEqual(followup_kind('faz uma história maior'),'longer')
        self.assertEqual(followup_kind('continua a história'),'continue')
        self.assertEqual(followup_kind('muda o final'),'ending')
        self.assertEqual(followup_kind('explica melhor'),'explain')
        for text in ('2+2', 'o que sabes fazer', 'conta-me uma história',
                     'o Python é uma linguagem de programação'):
            self.assertIsNone(followup_kind(text))

    def test_context_can_be_cleared(self):
        with tempfile.TemporaryDirectory() as root:
            ctx=DialogueMemory(Path(root))
            ctx.add('ola','Ola','test','dialogue')
            self.assertEqual(ctx.latest()['question'],'ola')
            ctx.clear()
            self.assertIsNone(ctx.latest())
            ctx.close()


if __name__=='__main__':unittest.main()

class ContextControls(unittest.TestCase):
    def test_story_pronouns_and_cli_clear(self):
        from nord_brain.cli import handle
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=Brain(Path(root),start_autonomy=False)
            b.core=NeuralCore(Path(root)/'brain-v2.pt',CONFIG)
            try:
                narrative,_=b.chat('conta-me uma historia')
                name=narrative.split('\n\n',1)[1].split(',',1)[0]
                who,_=b.chat('qual era o nome da personagem?')
                self.assertIn(name,who)
                self.assertIn('último assunto',handle(b,'/contexto').lower())
                self.assertIn('apagado',handle(b,'/limpar-contexto'))
                self.assertIsNone(b.dialogue.latest())
                self.assertIsNone(b.memory.exact('qual era o nome da personagem?'))
            finally:b.close()

class GeneralContextRegression(unittest.TestCase):
    def make_brain(self, root):
        b=Brain(Path(root),start_autonomy=False)
        b.core=NeuralCore(Path(root)/'brain-v2.pt',CONFIG)
        return b

    def test_math_becomes_context_for_non_math_discussion(self):
        with tempfile.TemporaryDirectory() as root, patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.make_brain(root)
            try:
                first,_=b.chat('4+4')
                self.assertIn('8',first)
                messages=b.dialogue.context_messages()
                self.assertEqual(messages[0]['role'],'user')
                self.assertEqual(messages[1]['role'],'assistant')
                self.assertIn('4+4',messages[0]['content'])
                self.assertIn('8',messages[1]['content'])
                follow,origin=b.chat('o que achas disso?')
                self.assertIn('4+4',follow)
                self.assertIn('continuidade contextual',origin)
                self.assertEqual(b.dialogue.latest()['topic'],'4+4')
                self.assertIsNone(b.memory.exact('o que achas disso?'))
            finally:b.close()

    def test_context_persists_across_restarts_with_bookmarks(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            b=self.make_brain(root)
            b.chat('4+4')
            b.chat('explica melhor')
            b.close()
            b=self.make_brain(root)
            try:
                self.assertEqual(b.dialogue.latest()['topic'],'4+4')
                self.assertEqual(len(b.dialogue.context_messages()),4)
                msg,origin=b.chat('o que sugeres?')
                self.assertIn('4+4',msg)
                self.assertIn('continuidade',origin)
                self.assertIsNone(b.memory.exact('explica melhor'))
            finally:b.close()

    def test_context_is_sent_to_local_llm_in_correct_roles_after_restart(self):
        import json
        class FakeResponse:
            status=200
            def __init__(self,data):self.data=data
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self,n=600001):return self.data[:n]
        class Opener:
            def __init__(self):self.messages=None
            def open(self,req,timeout):
                self.messages=json.loads(req.data)['messages']
                return FakeResponse(json.dumps({'choices':[{'message':{'content':'Essa soma é correta, e há outras maneiras de obter oito.'}}]}).encode())
        with tempfile.TemporaryDirectory() as root:
            with patch.dict('os.environ',{'NORD_LLM':'off'}):
                b=self.make_brain(root)
                b.chat('4+4')
                b.close()
            b=self.make_brain(root)
            try:
                opener=Opener()
                with patch.dict('os.environ',{'NORD_LLM':'on'}),patch(
                    'nord_brain.language_backend.request.build_opener',return_value=opener):
                    reply,_=b.chat('e o que achas disso?')
                self.assertIn('soma',reply)
                self.assertEqual([m['role'] for m in opener.messages],
                                 ['system','user','assistant','user'])
                self.assertIn('4+4',opener.messages[1]['content'])
                self.assertIn('8',opener.messages[2]['content'])
                self.assertIn('o que achas disso',opener.messages[3]['content'])
            finally:b.close()

    def test_arbitrary_new_questions_receive_previous_interactions(self):
        with tempfile.TemporaryDirectory() as root:
            b=self.make_brain(root)
            try:
                with patch.object(b.language,'generate',side_effect=[
                    'O motor converte energia em movimento.',
                    'Uma alternativa seria melhorar a eficiência do motor.']) as mock:
                    first,_=b.chat('Dá-me uma ideia para um motor')
                    second,_=b.chat('E como o podíamos melhorar?')
                self.assertIn('eficiência',second)
                messages=b.dialogue.context_messages()
                self.assertEqual(len(messages),4)
                self.assertEqual(messages[0]['content'],'Dá-me uma ideia para um motor')
                self.assertIn('energia',messages[1]['content'])
            finally:b.close()

    def test_old_dialogue_database_migrates_without_deleting_history(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'dialogue.sqlite3'
            conn=sqlite3.connect(path)
            conn.execute('CREATE TABLE turns (id INTEGER PRIMARY KEY, question TEXT, reply TEXT, origin TEXT, focus TEXT, created INTEGER)')
            conn.execute("INSERT INTO turns(question,reply,origin,focus,created) VALUES ('Como vai?','Bem.','chat','dialogue',1)")
            conn.commit();conn.close()
            d=DialogueMemory(Path(root))
            try:
                self.assertEqual(d.latest()['question'],'Como vai?')
                d.add('E agora?','Bem também.','chat','dialogue','Como vai?')
                self.assertEqual(d.latest()['topic'],'Como vai?')
            finally:d.close()

    def test_non_followups_not_mistaken_for_lessons(self):
        for q in ('conta-me uma história','10+2','e 10 + 2','o Python é uma linguagem de programação',
                  'quando eu perguntar X responde Y','o que sabes fazer'):
            self.assertIsNone(followup_kind(q),q)
