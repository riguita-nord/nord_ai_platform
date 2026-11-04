"""Persistent long-form stories with bounded CPU chapter generation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nord_brain.story_studio import StoryStudio, BookCommand, parse_book_command
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore, TransformerConfig
from nord_brain.cli import handle

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)


class Commands(unittest.TestCase):
    def test_natural_book_requests(self):
        cases=[
            ('Conta-me uma história longa sobre dragões','start',10,'dragões'),
            ('Conta-me uma hitoria longa com 12 capítulos sobre robôs','start',12,'robôs'),
            ('Escreve um livro sobre uma cidade perdida','start',10,'cidade perdida'),
            ('Cria um romance de 8 capítulos sobre uma viagem','start',8,'viagem'),
            ('continua a história','continue',1,''),
            ('escreve o próximo capítulo','continue',1,''),
            ('Mostra capítulo 2','read',2,''),
            ('Lista livros','list',1,''),
            ('Exporta o livro','export',1,''),
        ]
        for phrase,kind,count,topic in cases:
            with self.subTest(phrase=phrase):
                command=parse_book_command(phrase)
                self.assertIsNotNone(command)
                self.assertEqual(command.action,kind)
                self.assertEqual(command.amount,count)
                self.assertIn(topic,command.topic)
        self.assertIsNone(parse_book_command('Qual é a história de Portugal?'))
        self.assertIsNone(parse_book_command('Conta-me uma história curta'))

    def test_invalid_amounts_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            studio=StoryStudio(Path(root))
            try:
                for amount in (0,2,25):
                    with self.assertRaises(ValueError):studio.create('história espacial',amount)
                studio.create('história espacial',4)
                for amount in (0,4,11):
                    with self.assertRaises(ValueError):studio.continue_book(None,amount)
            finally:studio.close()


class Persistence(unittest.TestCase):
    def test_chapters_persist_and_export_in_order(self):
        with tempfile.TemporaryDirectory() as root:
            studio=StoryStudio(Path(root))
            book=studio.create('um robô que descobre o mundo',4)
            self.assertEqual(book['written'],0)
            result=studio.continue_book(None,amount=2)
            self.assertEqual([x['number'] for x in result['chapters']],[1,2])
            self.assertGreater(len(result['chapters'][0]['body']),1200)
            self.assertNotEqual(result['chapters'][0]['body'],result['chapters'][1]['body'])
            self.assertIn(book['bible']['protagonist'].split(',')[0],result['chapters'][0]['body'])
            studio.close()
            studio=StoryStudio(Path(root))
            self.assertEqual(studio.active()['written'],2)
            self.assertIsNotNone(studio.chapter(1))
            self.assertEqual(len(studio.list_books()),1)
            studio.continue_book(None,amount=3)
            self.assertEqual(studio.active()['written'],4)
            self.assertEqual(studio.continue_book(None)['chapters'],[])
            path=studio.export()
            self.assertEqual(path.name,'livro-001.md')
            text=path.read_text(encoding='utf-8')
            self.assertIn('Capítulo 4',text)
            self.assertLess(text.index('Capítulo 1'),text.index('Capítulo 4'))
            studio.close()

    def test_multiple_books_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as root:
            studio=StoryStudio(Path(root))
            first=studio.create('a cidade de vidro',3)
            studio.continue_book(None)
            second=studio.create('uma biblioteca oceânica',3)
            self.assertEqual(second['written'],0)
            self.assertEqual(studio.select(first['id'])['written'],1)
            self.assertEqual(len(studio.list_books()),2)
            studio.close()

    def test_local_model_receives_compact_continuity_without_chat_history(self):
        with tempfile.TemporaryDirectory() as root:
            studio=StoryStudio(Path(root))
            studio.create('uma viagem ao futuro',3)
            class Model:
                def __init__(self):self.prompts=[]
                def generate(self,prompt,*,creative,max_tokens,remember):
                    self.prompts.append((prompt,creative,max_tokens,remember))
                    return ('Uma porta fechou-se, e a protagonista encontrou uma pista nova. '
                            'Sabia que precisaria de escolher um caminho difícil. ')*5
            model=Model()
            studio.continue_book(model,2)
            self.assertEqual(len(model.prompts),2)
            self.assertIn('Últimas linhas anteriores:',model.prompts[1][0])
            self.assertIn('Capítulo 1',model.prompts[1][0])
            self.assertTrue(all(not row[3] for row in model.prompts))
            self.assertLess(len(model.prompts[1][0]),2800)
            studio.close()


class Runtime(unittest.TestCase):
    def test_story_commands_do_not_train_language_model(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'NORD_LLM':'off'}):
            brain=Brain(Path(root),start_autonomy=False)
            brain.core=NeuralCore(Path(root)/'brain-v2.pt',CONFIG)
            try:
                initial=brain.core.steps
                started,source=brain.chat('cria um livro sobre uma estação espacial')
                self.assertIn('Capítulo 1',started)
                self.assertIn('persistente',source)
                next_chapter,_=brain.chat('continua a história')
                self.assertIn('Capítulo 2',next_chapter)
                plan,_=brain.chat('plano do livro')
                self.assertIn('2/10',plan)
                self.assertEqual(brain.core.steps,initial)
                self.assertIsNone(brain.memory.exact('cria um livro sobre uma estação espacial'))
                export=handle(brain,'/livro exportar')
                self.assertIn('livro-001.md',export)
                self.assertIn('1',handle(brain,'/livro listar'))
                self.assertIn('Capítulo 1',handle(brain,'/livro ler 1'))
                self.assertIn('3',handle(brain,'/livro continuar'))
            finally:brain.close()
            again=Brain(Path(root),start_autonomy=False)
            again.core=NeuralCore(Path(root)/'brain-v2.pt',CONFIG)
            try:self.assertEqual(again.stories.active()['written'],3)
            finally:again.close()


if __name__=='__main__':unittest.main()
