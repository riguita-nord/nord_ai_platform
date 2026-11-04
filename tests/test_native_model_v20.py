"""Native Nord-language tests: no pretrained model and no remote calls."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import torch
from nord_brain.core import NeuralCore,TransformerConfig
from nord_brain.native_language import NativeLanguage,coherent_sample,_compact_prompt
from nord_brain.native_training import train_approved_file
from nord_brain.runtime import Brain

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)


class NativeLanguageTests(unittest.TestCase):
    def test_own_transformer_is_default_without_qwen(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{
                'NORD_LANGUAGE_BACKEND':'native'},clear=False):
            b=Brain(Path(root),start_autonomy=False)
            try:
                self.assertIsInstance(b.language,NativeLanguage)
                self.assertIn('pesos treináveis',b.language.status())
                self.assertNotIn('Qwen',b.language.__class__.__name__)
            finally:b.close()

    def test_generation_is_from_core_weights_and_dialogue_context(self):
        generated='Podemos continuar a desenvolver esta ideia, com novos caminhos e exemplos.'
        core=SimpleNamespace(steps=150,model=SimpleNamespace(
            config=SimpleNamespace(context=512)),generate=Mock(return_value=generated))
        brain=SimpleNamespace(core=core)
        lang=NativeLanguage(brain)
        called=[]
        lang.context_provider=lambda prompt:(called.append(prompt) or [
            {'role':'user','content':'Um robô que explora a cidade.'},
            {'role':'assistant','content':'Podemos testar sensores e memória.'}])
        with patch.dict('os.environ',{'NORD_NATIVE_MIN_STEPS':'1','NORD_LLM':''}):
            output=lang.generate('E podemos desenvolver isso?',creative=True)
        self.assertEqual(output,generated)
        self.assertEqual(called,['E podemos desenvolver isso?'])
        prompt=core.generate.call_args.args[0]
        self.assertIn('robô',prompt)
        self.assertIn('Pedido:',prompt)
        self.assertLessEqual(len(prompt.encode('utf-8')),172)

    def test_rejects_untrained_and_gibberish_outputs(self):
        brain=SimpleNamespace(core=SimpleNamespace(steps=0,model=SimpleNamespace(
            config=SimpleNamespace(context=256)),
            generate=Mock(return_value='aaaa aaaa aaaa aaaa')))
        lang=NativeLanguage(brain)
        with patch.dict('os.environ',{'NORD_LLM':'','NORD_NATIVE_MIN_STEPS':'80'}):
            self.assertIsNone(lang.generate('conta-me uma história'))
            brain.core.steps=100
            self.assertIsNone(lang.generate('conta-me uma história'))
        self.assertFalse(coherent_sample('t t t t t'))
        self.assertTrue(coherent_sample(
            'Podemos explorar esta ideia através de experiências novas, se quiseres.'))

    def test_explicit_off_disables_generation(self):
        brain=SimpleNamespace(core=SimpleNamespace(
            steps=1000,generate=Mock()))
        with patch.dict('os.environ',{'NORD_NATIVE_GENERATION':'off'}):
            self.assertIsNone(NativeLanguage(brain).generate('olá'))
        brain.core.generate.assert_not_called()


class CorpusTrainingTests(unittest.TestCase):
    def test_causal_next_byte_windows_and_persistent_weight_updates(self):
        with tempfile.TemporaryDirectory() as root:
            core=NeuralCore(Path(root)/'native.pt',CONFIG)
            windows=core._text_windows('Uma história bastante interessante.',128)
            ids,labels=windows[0]
            self.assertEqual(ids.shape,labels.shape)
            self.assertEqual(int(ids[0,0]),257)
            self.assertEqual(int(labels[0,0]),ord('U'))
            old=core.model.embed.weight.detach().clone()
            loss=core.train_text('Uma história bastante interessante. '*9,updates=2)
            self.assertGreater(loss,0)
            self.assertEqual(core.steps,2)
            self.assertFalse(torch.equal(old,core.model.embed.weight))
            core.save()
            new=NeuralCore(Path(root)/'native.pt')
            self.assertEqual(new.steps,2)

    def test_human_approved_file_is_only_text_training_source(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{
                'NORD_LANGUAGE_BACKEND':'native','NORD_LLM':'off'}):
            b=Brain(Path(root),start_autonomy=False)
            b.core=NeuralCore(Path(root)/'brain-v2.pt',CONFIG)
            filename=Path(root)/'historia.txt'
            filename.write_text(
                ('Num lugar pequeno, a personagem encontrou um segredo. '
                 'A descoberta alterou as escolhas e a vida dos amigos.\n')*5,
                encoding='utf-8')
            try:
                previous=b.core.steps
                result=train_approved_file(b,filename,updates=2)
                self.assertTrue(result.accepted)
                self.assertEqual(result.updates,2)
                self.assertEqual(b.core.steps,previous+2)
                self.assertEqual(b.memory.lesson_count(),0)
                self.assertEqual(NeuralCore(Path(root)/'brain-v2.pt').steps,b.core.steps)
                self.assertIn('historia.txt',result.path)
                with self.assertRaises(ValueError):
                    train_approved_file(b,Path(root)/'nao-existe.txt',updates=1)
            finally:b.close()


if __name__=='__main__':unittest.main()
