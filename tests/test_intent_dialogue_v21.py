"""Neural intent and supervised multi-turn European Portuguese regressions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nord_brain.core import NeuralCore,TransformerConfig
from nord_brain.runtime import Brain
from nord_brain.intent_learning import NeuralIntent,LABELS
from nord_brain.dialogue_training import curriculum,evaluate,train,describe
from nord_brain.cli import handle

SMALL=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,
                        intermediate_size=64,context=128)


class IntentLearningTests(unittest.TestCase):
    def test_trainable_classifier_survives_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            head=NeuralIntent(Path(folder))
            self.assertGreaterEqual(head.validation_accuracy,.60)
            self.assertTrue(head.path.exists())
            p1=head.predict('consegues por com um final melhor','story')
            self.assertEqual(p1[0],'ending',p1)
            p2=head.predict('divide essa história em capítulos','story')
            self.assertIn(p2[0],('chapters',None),p2)
            self.assertIsNone(head.predict('2+2','story')[0])
            second=NeuralIntent(Path(folder))
            self.assertEqual(second.predict(
                'consegues por com um final melhor','story')[0],'ending')
            self.assertEqual(head.validation_accuracy,second.validation_accuracy)

    def test_uses_context_to_revise_existing_story_not_teacher_memory(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(
                'os.environ',{'NORD_LANGUAGE_BACKEND':'native',
                              'NORD_NATIVE_GENERATION':'off'}):
            b=Brain(Path(folder),start_autonomy=False)
            b.core=NeuralCore(Path(folder)/'brain-v2.pt',SMALL)
            try:
                original,_=b.chat('conta-me uma historia')
                extended,_=b.chat('faz mais extenso')
                steps=b.core.steps
                revised,origin=b.chat('consegues por com um final melhor')
                self.assertIn('ficção contextual',origin)
                self.assertNotIn('Ainda não sei responder',revised)
                self.assertIn(original.split('\n\n',1)[1].split(',',1)[0],revised)
                self.assertNotEqual(revised,extended)
                self.assertEqual(b.core.steps,steps)
                self.assertFalse(b.awaiting_explanation)
                self.assertIsNone(b.memory.exact('consegues por com um final melhor'))
                self.assertEqual(b.dialogue.latest()['focus'],'story')
            finally:b.close()

    def test_explicit_new_topic_and_teacher_fact_are_not_revisions(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(
                'os.environ',{'NORD_NATIVE_GENERATION':'off'}):
            b=Brain(Path(folder),start_autonomy=False)
            b.core=NeuralCore(Path(folder)/'brain-v2.pt',SMALL)
            try:
                b.chat('conta-me uma historia')
                answer,origin=b.chat('2+2')
                self.assertIn('4',answer)
                self.assertEqual(b.dialogue.latest()['focus'],'math')
                b.chat('a luz é uma forma de energia')
                self.assertIsNotNone(b.memory.lookup_concept('luz'))
            finally:b.close()


class DialogueTrainingTests(unittest.TestCase):
    def brain(self,folder):
        b=Brain(Path(folder),start_autonomy=False)
        b.core=NeuralCore(Path(folder)/'brain-v2.pt',SMALL)
        return b

    def test_contextual_holdouts_disjoint(self):
        training,holdout=curriculum()
        self.assertEqual(len(training),180)
        self.assertEqual(len(holdout),60)
        self.assertFalse(set(x for x,_ in training)&set(x for x,_ in holdout))
        self.assertTrue(all('Nord:' in q and 'Tu:' in q for q,_ in training))

    def test_cpu_weights_change_or_rollback_if_validation_worsens(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.brain(folder)
            try:
                before=b.core.steps
                metrics=evaluate(b)
                self.assertEqual(metrics['heldout_dialogues'],len(curriculum()[1]))
                self.assertEqual(metrics['legacy_holdouts'],18)
                result=train(b,updates=2)
                self.assertIn('aceite' if result['accepted'] else 'rejeitado',
                              describe(result))
                self.assertEqual(b.core.steps,
                                 before+result['updates'])
                if result['accepted']:
                    self.assertGreaterEqual(result['updates'],2)
                else:
                    self.assertEqual(result['updates'],0)
                self.assertEqual(b.memory.teacher_lesson_count(),0)
                self.assertIsNone(b.memory.exact('podes desenvolver melhor'))
                self.assertIn('Diálogos com contexto',handle(b,'/avaliar-dialogos'))
            finally:b.close()

    def test_respects_teacher_pause(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.brain(folder)
            try:
                b.memory.pause_autonomy(True)
                with self.assertRaises(ValueError):
                    train(b,1)
            finally:b.close()


if __name__=='__main__':
    unittest.main()
