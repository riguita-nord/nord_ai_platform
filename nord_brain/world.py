"""CPU-only learned predictive world model for a virtual machine.

A small neural network learns *action consequences* from interactions in grid
worlds. It receives only a local observation and a candidate action, predicts
movement and collision, compares all four actions, and receives feedback from
executing the chosen action. Crucially, the simulator is an environment, not
an oracle supplied as input to the predictor or planner.

This is an experiment in model-based control. It does not provide robotics
safety, visual perception, language intelligence, or a trained Qwen model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTION_NAMES = ('norte', 'este', 'sul', 'oeste')


@dataclass(frozen=True)
class Experience:
    observed: tuple[float, ...]
    action: int
    actual: tuple[float, float, float]  # delta_x, delta_y, collision (0/1)


class GridWorld:
    """Deterministic seeded environment, with a reachable goal and local sensors.

    Ground-truth walls and destination are used only by the simulator; policy
    and neural world model see local wall sensors plus target displacement.
    """
    def __init__(self, seed: int, size: int = 7, density: float = .13):
        if not 5 <= size <= 11 or not 0 <= density <= .35:
            raise ValueError('Simulação fora dos limites definidos')
        self.size = size
        rng = random.Random(seed)
        self.position = (rng.randrange(size), rng.randrange(size))
        self.target = self.position
        self.obstacles: frozenset[tuple[int, int]] = frozenset()
        for _ in range(80):
            target = (rng.randrange(size), rng.randrange(size))
            if target == self.position:
                continue
            obs = frozenset((x, y) for x in range(size) for y in range(size)
                            if (x, y) not in (self.position, target)
                            and rng.random() < density)
            if self._reachable(self.position, target, obs):
                self.target, self.obstacles = target, obs
                break
        else:
            self.target = ((self.position[0] + 3) % size, (self.position[1] + 2) % size)

    def _valid(self, pos: tuple[int, int]) -> bool:
        return (0 <= pos[0] < self.size and 0 <= pos[1] < self.size
                and pos not in self.obstacles)

    def _reachable(self, start, target, obstacles) -> bool:
        queue = [start]
        seen = {start}
        for x, y in queue:
            if (x, y) == target:
                return True
            for dx, dy in DIRECTIONS:
                nxt = (x + dx, y + dy)
                if (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size
                        and nxt not in obstacles and nxt not in seen):
                    queue.append(nxt)
                    seen.add(nxt)
        return False

    def sense(self) -> tuple[float, ...]:
        px, py = self.position
        dx, dy = self.target[0] - px, self.target[1] - py
        nearby = tuple(float(not self._valid((px + sx, py + sy)))
                       for sx, sy in DIRECTIONS)
        return (dx / self.size, dy / self.size) + nearby

    def move(self, action: int) -> tuple[float, float, float]:
        if action not in range(4):
            raise ValueError('Ação desconhecida')
        px, py = self.position
        dx, dy = DIRECTIONS[action]
        nxt = (px + dx, py + dy)
        collision = not self._valid(nxt)
        if not collision:
            self.position = nxt
        return (float(self.position[0] - px), float(self.position[1] - py), float(collision))

    @property
    def reached(self) -> bool:
        return self.position == self.target


class PredictiveModel(nn.Module):
    """10 inputs: goal relative (2), adjacent obstacles (4), action (one-hot 4).

    Output: predicted x/y displacement and collision logit.
    Tiny enough to train on CPU; it is NOT a language model.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 48), nn.Tanh(), nn.Linear(48, 48),
                                 nn.Tanh(), nn.Linear(48, 3))

    def forward(self, features):
        return self.net(features)


def encode(observation: tuple[float, ...], action: int) -> tuple[float, ...]:
    if len(observation) != 6 or action not in range(4):
        raise ValueError('Observação ou ação inválida')
    return tuple(float(x) for x in observation) + tuple(float(i == action) for i in range(4))


def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Both physical movement and collision prediction are supervised from real
    # simulator observations, not from the model's own generated text.
    return F.mse_loss(torch.tanh(pred[:, :2]), target[:, :2]) + .35 * F.binary_cross_entropy_with_logits(pred[:, 2], target[:, 2])


class WorldMind:
    """Persistent CPU world-model that compares and tests action hypotheses."""
    def __init__(self, path: Path, seed: int = 2077):
        self.path = Path(path)
        self.model = PredictiveModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.006)
        self._seed = seed
        self.observations = 0
        self.train_steps = 0
        self.episodes = 0
        self.inputs = torch.empty((0, 10), dtype=torch.float32)
        self.targets = torch.empty((0, 3), dtype=torch.float32)
        if self.path.exists():
            state = torch.load(self.path, map_location='cpu', weights_only=True)
            if int(state.get('version', -1)) != 1:
                raise ValueError('Checkpoint do mundo virtual incompatível')
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.observations = int(state['observations'])
            self.train_steps = int(state['train_steps'])
            self.episodes = int(state['episodes'])
            self.inputs = state['inputs'].float().reshape(-1, 10)
            self.targets = state['targets'].float().reshape(-1, 3)
        self.model.eval()

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.nord-world-', suffix='.pt', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                torch.save({'version': 1, 'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'observations': self.observations,
                            'train_steps': self.train_steps, 'episodes': self.episodes,
                            'inputs': self.inputs, 'targets': self.targets}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def observe(self, observation: tuple[float, ...], action: int,
                consequence: tuple[float, float, float]):
        x = torch.tensor([encode(observation, action)], dtype=torch.float32)
        y = torch.tensor([consequence], dtype=torch.float32)
        self.inputs = torch.cat((self.inputs, x), dim=0)[-4096:]
        self.targets = torch.cat((self.targets, y), dim=0)[-4096:]
        self.observations += 1

    @torch.inference_mode()
    def hypothesize(self, observation: tuple[float, ...]):
        """Imagine all four available actions using learned model weights."""
        features = torch.tensor([encode(observation, a) for a in range(4)],dtype=torch.float32)
        self.model.eval()
        predictions = self.model(features)
        moves = torch.tanh(predictions[:, :2]).tolist()
        collisions = torch.sigmoid(predictions[:, 2]).tolist()
        goal_x = observation[0] * 7.0
        goal_y = observation[1] * 7.0
        base_dist = abs(goal_x) + abs(goal_y)
        result = []
        for action in range(4):
            dx, dy = moves[action]
            # Score is an explicit task objective; physical consequences are
            # learned, not hardcoded from the action name.
            progress = base_dist - abs(goal_x - dx) - abs(goal_y - dy)
            risk = collisions[action]
            result.append({'action': action, 'name': ACTION_NAMES[action],
                           'delta': (dx, dy), 'collision_risk': risk,
                           'score': progress - 1.5 * risk})
        return result

    def choose(self, observation: tuple[float, ...], rng: random.Random,
               explore: float = .15):
        candidates = self.hypothesize(observation)
        if rng.random() < explore:
            return rng.randrange(4), candidates
        return max(candidates,key=lambda h:h['score'])['action'], candidates

    def train(self, updates: int = 16, batch_size: int = 48) -> float | None:
        if len(self.inputs) < 12:
            return None
        self.model.train()
        loss_value = None
        for _ in range(max(1,min(updates,400))):
            index = torch.randint(len(self.inputs), (min(batch_size,len(self.inputs)),))
            predicted = self.model(self.inputs[index])
            loss = _loss(predicted, self.targets[index])
            if not torch.isfinite(loss):
                raise FloatingPointError('Perda do mundo virtual inválida')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.train_steps += 1
            loss_value = float(loss.detach())
        self.model.eval()
        return loss_value

    @staticmethod
    def _heldout_examples():
        """Unseen world seeds (never used for autonomous experience collection)."""
        rows=[]
        for i in range(60):
            world=GridWorld(100000+i, size=7, density=.08+(.03*(i%4)))
            rng=random.Random(300000+i)
            for step in range(3):
                observation=world.sense()
                for action in range(4):
                    # Evaluate all counterfactual actions from the SAME state.
                    dx,dy=DIRECTIONS[action]
                    px,py=world.position
                    blocked=not world._valid((px+dx,py+dy))
                    actual=(0.,0.,1.) if blocked else (float(dx),float(dy),0.)
                    rows.append((encode(observation,action),actual))
                world.move(rng.randrange(4))
        x=torch.tensor([a for a,_ in rows],dtype=torch.float32)
        y=torch.tensor([b for _,b in rows],dtype=torch.float32)
        return x,y

    @torch.inference_mode()
    def evaluate(self) -> dict:
        """Measurements on different world seeds; NOT chat or general intelligence."""
        x,y=self._heldout_examples()
        self.model.eval()
        raw=self.model(x)
        predicted=torch.tanh(raw[:,:2])
        collision=torch.sigmoid(raw[:,2])
        return {'movement_mse': round(float(F.mse_loss(predicted,y[:,:2])),5),
                'collision_accuracy': round(float(((collision>=.5)==(y[:,2]>=.5)).float().mean()),4),
                'joint_loss': round(float(_loss(raw,y)),5)}

    def experience_cycle(self, scenarios: int = 3, steps_per_world: int = 12,
                         updates: int = 16, save: bool = True) -> dict:
        """Run, test, learn. Executes ONLY in the bounded virtual environment."""
        if not 1 <= scenarios <= 20 or not 1 <= steps_per_world <= 40:
            raise ValueError('Número de experiências fora dos limites')
        before = self.evaluate()
        first_hypotheses = None
        first_actual = None
        successes = 0
        collisions = 0
        for index in range(scenarios):
            episode_id=self.episodes+index
            world=GridWorld(seed=1000+episode_id, size=7,
                            density=.10+.035*(episode_id%4))
            rng=random.Random(7345+episode_id)
            for _ in range(steps_per_world):
                if world.reached:
                    break
                observation=world.sense()
                # In early training, explore often. Otherwise weigh predicted
                # consequences but keep some exploration for new situations.
                eps=.65 if self.observations<128 else .22
                action,hypotheses=self.choose(observation,rng,explore=eps)
                actual=world.move(action)
                if first_hypotheses is None:
                    first_hypotheses=hypotheses
                    first_actual={'action':ACTION_NAMES[action], 'delta':actual[:2],
                                  'collision':bool(actual[2])}
                collisions+=int(actual[2])
                self.observe(observation,action,actual)
            successes+=int(world.reached)
        self.episodes+=scenarios
        loss=self.train(updates=updates)
        after=self.evaluate()
        if save:
            self.save()
        return {'scenarios':scenarios, 'goals_reached':successes,
                'collisions':collisions, 'observations':self.observations,
                'train_steps':self.train_steps, 'train_loss':loss,
                'before':before, 'after':after,
                'hypotheses':first_hypotheses or [], 'actual':first_actual}

    def describe_cycle(self, result: dict) -> str:
        lines=[f"Experimentei {result['scenarios']} cenários virtuais, com ações e consequências reais no simulador.",
               'Antes de agir comparei quatro hipóteses (valores previstos pelos meus pesos):']
        for h in result['hypotheses']:
            dx,dy=h['delta']
            lines.append(f"  • {h['name']}: movimento previsto ({dx:+.2f}, {dy:+.2f}); "
                         f"risco previsto de colisão {h['collision_risk']:.0%}; valor {h['score']:+.2f}")
        if result['actual']:
            event=result['actual']
            lines.append(f"Executei {event['action']}; observei movimento {event['delta']} "
                         f"e {'colisão' if event['collision'] else 'sem colisão'}.")
        lines.extend([
            f"Resultado: {result['goals_reached']}/{result['scenarios']} objetivos alcançados; "
            f"{result['collisions']} colisões tentadas.",
            f"Treino: {result['observations']} experiências e {result['train_steps']} atualizações de pesos.",
            f"Erro de previsão em mundos não usados para treino: "
            f"{result['before']['joint_loss']:.4f} → {result['after']['joint_loss']:.4f} "
            '(menor é melhor, mas pode oscilar).',
            'Isto mede aprendizagem do modelo do mundo, não inteligência geral nem controlo de um robô real.'
        ])
        return '\n'.join(lines)
