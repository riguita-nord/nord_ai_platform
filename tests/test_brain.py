import tempfile
import unittest
from pathlib import Path
import torch

from nord_brain.core import NeuralCore, NordTransformer, TransformerConfig
from nord_brain.runtime import Brain
from nord_brain.cli import handle

# Small configuration for tests, not an intelligence benchmark.
TEST_CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)


class TransformerTests(unittest.TestCase):
    def test_causal_attention(self):
        model = NordTransformer(TEST_CONFIG)
        model.eval()
        with torch.no_grad():
            a = model(torch.tensor([[1,2,3,4]]))[0,:3]
            b = model(torch.tensor([[1,2,3,99]]))[0,:3]
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

    def test_train_changes_weights_and_persists(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.pt'
            core = NeuralCore(path, TEST_CONFIG)
            old = core.model.embed.weight.detach().clone()
            loss = core.train('olá','olá!',epochs=2)
            self.assertTrue(0 < loss < 100)
            self.assertFalse(torch.equal(old,core.model.embed.weight))
            self.assertEqual(core.steps,2)
            core.save()
            loaded = NeuralCore(path)
            self.assertEqual(loaded.steps,2)
            self.assertEqual(loaded.parameter_count,core.parameter_count)
            torch.testing.assert_close(loaded.model.embed.weight,core.model.embed.weight)
            self.assertIn('state',loaded.optimizer.state_dict())

    def test_growth_preserves_logits(self):
        with tempfile.TemporaryDirectory() as temp:
            core = NeuralCore(Path(temp)/'model.pt',TEST_CONFIG)
            core.train('olá','sim',epochs=1)
            data = torch.tensor([[1,2,3,4,5]])
            core.model.eval()
            with torch.no_grad():
                before = core.model(data).clone()
            old = core.parameter_count
            core.grow(16)
            with torch.no_grad():
                after = core.model(data)
            self.assertGreater(core.parameter_count,old)
            self.assertEqual(core.hidden_size,80)
            torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-6)
            core.train('olá','sim',epochs=1)
            core.save()
            self.assertEqual(NeuralCore(core.path).hidden_size,80)

    def test_only_answer_is_training_target(self):
        ids, labels = NeuralCore._example('a','b',128)
        prompt = 'Pergunta: a\nResposta: '.encode()
        self.assertEqual(ids.shape,labels.shape)
        self.assertTrue(torch.all(labels[0,:len(prompt)] == -100))
        self.assertEqual(int(labels[0,len(prompt)]),ord('b'))

    def test_training_loss_can_decrease(self):
        with tempfile.TemporaryDirectory() as temp:
            core=NeuralCore(Path(temp)/'learning.pt',TEST_CONFIG)
            first=core.train('olá', 'olá!', epochs=1)
            last=first
            for _ in range(14):
                last=core.train('olá', 'olá!', epochs=1)
            self.assertLess(last,first)

    def test_memory_is_not_claimed_as_neural_reasoning(self):
        with tempfile.TemporaryDirectory() as temp:
            b=Brain(Path(temp))
            # Default small model would be slow here, swap to test model first.
            b.core = NeuralCore(Path(temp)/'brain-v2.pt',TEST_CONFIG)
            handle(b,'/ensinar olá => bom dia')
            answer,origin = b.answer('olá')
            self.assertEqual(answer,'bom dia')
            self.assertIn('memória',origin)
            handle(b,'/teste quanto é 1+1 => dois')
            self.assertEqual(b.memory.lesson_count(),1)
            self.assertEqual(len(b.memory.tests()),1)
            self.assertTrue(b.vocabulary)
            with self.assertRaises(ValueError):
                handle(b,'/ensinar quanto é 1+1 => dois')
            with self.assertRaises(ValueError):
                handle(b,'/teste olá => bom dia')
            b.close()
            b2=Brain(Path(temp))
            self.assertEqual(b2.answer('olá')[0],'bom dia')
            self.assertGreater(b2.core.steps,0)
            b2.close()


if __name__=='__main__':
    unittest.main()
