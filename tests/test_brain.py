import tempfile
import unittest
from pathlib import Path
import numpy as np
from nord_brain.core import NeuralCore
from nord_brain.runtime import Brain
from nord_brain.cli import handle

class BrainTests(unittest.TestCase):
    def test_weights_change_and_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'weights.npz'
            n=NeuralCore(path,hidden=32,embedding=16)
            before=n.params['Why'].copy()
            loss=n.train('olá','olá!',epochs=2)
            self.assertTrue(np.isfinite(loss))
            self.assertFalse(np.array_equal(before,n.params['Why']))
            n.save()
            loaded=NeuralCore(path)
            self.assertEqual(loaded.steps,2)
            np.testing.assert_allclose(loaded.params['Why'],n.params['Why'])

    def test_growth_preserves_output(self):
        with tempfile.TemporaryDirectory() as temp:
            n=NeuralCore(Path(temp)/'weights.npz',hidden=32,embedding=16)
            n.train('abc','xyz',epochs=1)
            data=np.frombuffer(b'Pergunta: abc\nResposta: xyz',dtype=np.uint8)
            _,prob_before=n._forward(data)
            old=n.parameter_count
            n.grow(8)
            _,prob_after=n._forward(data)
            self.assertEqual(n.hidden_size,40)
            self.assertGreater(n.parameter_count,old)
            for a,b in zip(prob_before,prob_after):
                np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
            n.train('outra','resposta',epochs=1)
            n.save()
            self.assertEqual(NeuralCore(n.path).hidden_size,40)

    def test_memory_and_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            b=Brain(Path(temp))
            handle(b,'/ensinar olá => bom dia')
            self.assertEqual(b.answer('olá')[0],'bom dia')
            self.assertIn('memória',b.answer('olá')[1])
            handle(b,'/teste quanto é 1+1 => dois')
            self.assertEqual(b.memory.lesson_count(),1)
            self.assertEqual(len(b.memory.tests()),1)
            self.assertTrue(b.vocabulary)
            b.close()
            b2=Brain(Path(temp))
            self.assertEqual(b2.answer('olá')[0],'bom dia')
            self.assertGreater(b2.core.steps,0)
            b2.close()

if __name__=='__main__':unittest.main()
