"""Context-sensitive Portuguese follow-ups must never become false lessons."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nord_brain.creative import story
from nord_brain.dialogue import DialogueMemory, followup_kind
from nord_brain.story_studio import StoryStudio


class EllipsisParsingTests(unittest.TestCase):
    def test_elliptical_revision_requests(self):
        examples={
            'com capítulos e etc':'chapters',
            'em capítulos':'chapters',
            'organiza em capítulos':'chapters',
            'faz isso com capítulos':'chapters',
            'com 8 capítulos':'chapters',
            'sem spoilers':'refine',
            'com mais diálogos':'refine',
            'em pontos':'refine',
            'faz isso com exemplos':'refine',
            'por tópicos':'refine',
        }
        for phrase,expected in examples.items():
            self.assertEqual(followup_kind(phrase),expected,phrase)

    def test_dont_swallow_new_questions_or_teacher_facts(self):
        for phrase in ('quem é Elon Musk?', 'com quem falaste?',
                       'em que ano nasceu?', 'o Python é uma linguagem',
                       '2+2', 'e 2+2', 'conta-me uma história'):
            self.assertIsNone(followup_kind(phrase),phrase)


class ExistingStoryBookTests(unittest.TestCase):
    def test_adopts_previous_story_not_new_random_characters(self):
        with tempfile.TemporaryDirectory() as root:
            current=story('um enigma com uma carta')
            studio=StoryStudio(Path(root))
            try:
                result=studio.adopt_story(current,chapters=10)
                self.assertEqual(result['book']['written'],3)
                self.assertEqual(result['book']['planned'],10)
                self.assertEqual(len(studio.list_books()),1)
                self.assertEqual(result['book']['title'],current.splitlines()[0])
                text='\n\n'.join(ch['body'] for ch in result['chapters'])
                self.assertEqual(text,current.split('\n\n',1)[1])
                self.assertIn(result['book']['bible']['protagonist'],current)
                self.assertEqual(studio.chapter(1)[1],result['chapters'][0]['body'])
            finally:
                studio.close()
            reopened=StoryStudio(Path(root))
            try:
                self.assertEqual(reopened.active()['written'],3)
                continued=reopened.continue_book(None,1)
                self.assertEqual(continued['chapters'][0]['number'],4)
                self.assertIn('Capítulo',reopened._prompt(reopened.active(),5,'Nova pista','Avançar'))
            finally:
                reopened.close()

    def test_rejects_nonstory_and_does_not_store_false_teacher_facts(self):
        with tempfile.TemporaryDirectory() as root:
            studio=StoryStudio(Path(root))
            try:
                with self.assertRaises(ValueError):
                    studio.adopt_story('2 + 2 = 4')
                self.assertEqual(studio.list_books(),[])
            finally:
                studio.close()

    def test_gender_matches_selected_character(self):
        from nord_brain.creative import story as tell
        class Fixed:
            def choice(self,items):
                return items[-1] if 'músico' in ' '.join(str(v) for v in items) else items[0]
        # Check a deterministic generation with the actual selected name.
        for seed in range(20):
            import random
            text=tell('uma descoberta',rng=random.Random(seed))
            if text.split('\n\n',1)[1].startswith('Nuno,'):
                self.assertIn('Desde pequeno,',text)
                break
        else:
            self.fail('Nuno did not appear in deterministic story samples')


if __name__=='__main__':
    unittest.main()
