"""Small byte-level recurrent neural network with real online BPTT training.

This is a research model, not a pretrained language model. Predictions may be
incoherent, even after thousands of examples. NumPy only, CPU friendly.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import numpy as np


class NeuralCore:
    def __init__(self, path: Path, hidden: int = 96, embedding: int = 32, seed: int = 17):
        self.path = Path(path)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.params = {}
        if self.path.exists():
            self.load()
        else:
            self._initialize(hidden, embedding)

    def _initialize(self, hidden: int, embedding: int):
        if not 16 <= hidden <= 512 or not 8 <= embedding <= 128:
            raise ValueError('Dimensões neuronais fora dos limites')
        self.params = {
            'E': self.rng.normal(0, .04, (256, embedding)).astype('f4'),
            'Wxh': self.rng.normal(0, .08, (embedding, hidden)).astype('f4'),
            'Whh': self.rng.normal(0, .04, (hidden, hidden)).astype('f4'),
            'bh': np.zeros(hidden, dtype='f4'),
            'Why': self.rng.normal(0, .02, (hidden, 256)).astype('f4'),
            'by': np.zeros(256, dtype='f4'),
        }
        self.acc = {k: np.zeros_like(v) for k, v in self.params.items()}

    @property
    def hidden_size(self) -> int:
        return int(self.params['bh'].size)

    @property
    def parameter_count(self) -> int:
        return sum(v.size for v in self.params.values())

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        contents = {f'p_{k}': v for k, v in self.params.items()}
        contents.update({f'a_{k}': v for k, v in self.acc.items()})
        contents['steps'] = np.asarray([self.steps], dtype=np.int64)
        fd, temp = tempfile.mkstemp(dir=self.path.parent, prefix='.brain-', suffix='.npz')
        try:
            with os.fdopen(fd, 'wb') as handle:
                np.savez_compressed(handle, **contents)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def load(self):
        with np.load(self.path, allow_pickle=False) as data:
            keys = ('E', 'Wxh', 'Whh', 'bh', 'Why', 'by')
            self.params = {k: np.array(data['p_' + k], dtype='f4', copy=True) for k in keys}
            self.acc = {k: np.array(data['a_' + k], dtype='f4', copy=True) for k in keys}
            self.steps = int(data['steps'][0])
        h = self.hidden_size
        e = self.params['E'].shape[1]
        expected = {'E':(256,e), 'Wxh':(e,h), 'Whh':(h,h), 'bh':(h,), 'Why':(h,256), 'by':(256,)}
        if any(self.params[k].shape != shape or self.acc[k].shape != shape for k,shape in expected.items()):
            raise ValueError('Checkpoint neuronal incompatível')

    def _forward(self, data: np.ndarray):
        p = self.params
        h = np.zeros(self.hidden_size, dtype='f4')
        states = [h]
        probs = []
        for idx in data[:-1]:
            h = np.tanh(p['E'][idx] @ p['Wxh'] + h @ p['Whh'] + p['bh'])
            logits = h @ p['Why'] + p['by']
            logits = logits - logits.max()
            exps = np.exp(logits)
            probs.append(exps / exps.sum())
            states.append(h)
        return states, probs

    def train(self, prompt: str, answer: str, epochs: int = 2, rate: float = .015) -> float:
        if not 1 <= epochs <= 25:
            raise ValueError('Épocas fora dos limites')
        if not 0 < rate <= .1:
            raise ValueError('Taxa de aprendizagem inválida')
        # Train on a bounded segment, including both context and answer.
        sequence = ('Pergunta: ' + prompt + '\nResposta: ' + answer + '\n').encode('utf-8')
        data = np.frombuffer(sequence[:384], dtype=np.uint8)
        if data.size < 5:
            raise ValueError('Exemplo demasiado curto')
        loss = float('inf')
        p = self.params
        for _ in range(epochs):
            states, probs = self._forward(data)
            target = data[1:]
            loss = float(-sum(np.log(max(float(prob[int(y)]),1e-12)) for prob,y in zip(probs,target)) / len(target))
            grad = {k: np.zeros_like(v) for k,v in p.items()}
            dhnext = np.zeros(self.hidden_size, dtype='f4')
            for t in range(len(target)-1, -1, -1):
                dy = probs[t].copy()
                dy[target[t]] -= 1
                dy /= len(target)
                grad['Why'] += np.outer(states[t+1], dy)
                grad['by'] += dy
                dh = dy @ p['Why'].T + dhnext
                dz = dh * (1 - states[t+1] ** 2)
                grad['bh'] += dz
                grad['Wxh'] += np.outer(p['E'][data[t]], dz)
                grad['Whh'] += np.outer(states[t], dz)
                grad['E'][data[t]] += dz @ p['Wxh'].T
                dhnext = dz @ p['Whh'].T
            norm = np.sqrt(sum(float(np.sum(g*g)) for g in grad.values()))
            scale = min(1.0, 1.0 / (norm + 1e-8))
            for k in p:
                g = grad[k] * scale
                self.acc[k] += g*g
                p[k] -= rate * g / np.sqrt(self.acc[k] + 1e-5)
                if not np.isfinite(p[k]).all():
                    raise FloatingPointError('Treino neuronal instável; restaurar checkpoint')
            self.steps += 1
        return loss

    def generate(self, prompt: str, max_bytes: int = 140, temperature: float = .75) -> str:
        p = self.params
        prefix = ('Pergunta: ' + prompt + '\nResposta: ').encode('utf-8')[-512:]
        h = np.zeros(self.hidden_size, dtype='f4')
        for idx in prefix:
            h = np.tanh(p['E'][idx] @ p['Wxh'] + h @ p['Whh'] + p['bh'])
        result = bytearray()
        for _ in range(max(1, min(300, max_bytes))):
            logits = (h @ p['Why'] + p['by']) / max(.2, temperature)
            # Enforce simple byte-level sampling with a bounded top-k.
            choices = np.argpartition(logits, -16)[-16:]
            weights = np.exp(logits[choices]-logits[choices].max())
            weights /= weights.sum()
            idx = int(self.rng.choice(choices, p=weights))
            if idx == 10:
                break
            result.append(idx)
            h = np.tanh(p['E'][idx] @ p['Wxh'] + h @ p['Whh'] + p['bh'])
        return result.decode('utf-8', errors='ignore').strip()

    def grow(self, extra: int = 16) -> int:
        """Function-preserving expansion: new hidden channels have zero output effect.

        The new channels are activated internally and can learn through the new
        output weights. This does NOT improve intelligence by itself.
        """
        if extra < 1 or extra > 64 or self.hidden_size + extra > 512:
            raise ValueError('Crescimento fora dos limites (máximo 512 neurónios)')
        p, acc = self.params, self.acc
        old = self.hidden_size
        new = old + extra
        next_p = {}
        next_p['E'] = p['E'].copy()
        next_p['Wxh'] = np.concatenate([p['Wxh'], self.rng.normal(0,.02,(p['Wxh'].shape[0],extra)).astype('f4')],axis=1)
        next_p['Whh'] = np.zeros((new,new),dtype='f4')
        next_p['Whh'][:old,:old] = p['Whh']
        next_p['Whh'][:old,old:] = self.rng.normal(0,.02,(old,extra)).astype('f4')
        next_p['bh'] = np.concatenate([p['bh'],np.zeros(extra,dtype='f4')])
        next_p['Why'] = np.concatenate([p['Why'],np.zeros((extra,256),dtype='f4')],axis=0)
        next_p['by'] = p['by'].copy()
        self.params = next_p
        next_a = {k:np.zeros_like(v) for k,v in next_p.items()}
        for k in ('E','by'):
            next_a[k][...] = acc[k]
        next_a['Wxh'][:,:old] = acc['Wxh']
        next_a['Whh'][:old,:old] = acc['Whh']
        next_a['bh'][:old] = acc['bh']
        next_a['Why'][:old,:] = acc['Why']
        self.acc = next_a
        return self.hidden_size
