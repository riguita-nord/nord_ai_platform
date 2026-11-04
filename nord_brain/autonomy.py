"""Bounded autonomous replay, verifiable tasks, cautious growth evaluation.

No self-generated label is trusted. Internet extracts are unverified and never
become neural training targets automatically. Hardware budgets are enforced.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
import threading
import time

from .experiments import curriculum, MAX_LEVEL
from .internet import search
from .reasoning import math_question


def on(name: str, default: bool=False) -> bool:
    return os.getenv(name, 'on' if default else 'off').strip().lower() in ('1','on','true','yes')


def integer(name: str, default: int, lo: int, hi: int) -> int:
    try:return max(lo,min(hi,int(os.getenv(name,str(default)))))
    except ValueError:return default


def growth_candidate(scores: list[float], steps: int, lesson_count: int,
                     holdout_count: int, params: int, width: int,
                     max_params: int, max_width: int) -> bool:
    """A plateau suggests an experiment, never automatically proves capacity shortage."""
    if (len(scores)<3 or steps<40 or lesson_count<12 or holdout_count<6
            or params>=max_params or width+16>max_width):
        return False
    if not all(math.isfinite(v) for v in scores[-3:]):return False
    if scores[-1]<1.25:return False
    return min(scores[-3:]) > scores[0]*.96  # less than 4% improvement


@dataclass
class AutoSettings:
    interval: int = 30
    training: bool = True
    experiments: bool = True
    growth: bool = True
    research: bool = False
    max_params: int = 3000000
    max_width: int = 1024
    examples_per_tick: int = 4
    world_simulations: bool = True
    world_scenarios: int = 3
    hypotheses_per_tick: int = 3

    @classmethod
    def from_env(cls):
        return cls(interval=integer('NORD_AUTO_INTERVAL',30,5,3600),
                   training=on('NORD_AUTO_TRAIN',True),
                   experiments=on('NORD_AUTO_EXPERIMENTS',True),
                   growth=on('NORD_AUTO_GROW',True),
                   research=on('NORD_AUTONOMOUS_RESEARCH',True) and on('NORD_INTERNET',True),
                   max_params=integer('NORD_MAX_PARAMS',3000000,100000,10000000),
                   max_width=integer('NORD_MAX_MLP_WIDTH',1024,128,4096),
                   examples_per_tick=integer('NORD_MATH_EXAMPLES_PER_TICK',4,1,12),
                   world_simulations=on('NORD_WORLD_SIM',True),
                   world_scenarios=integer('NORD_WORLD_SCENARIOS',3,1,10),
                   hypotheses_per_tick=integer('NORD_HYPOTHESES_PER_TICK',3,1,8))


class Autonomy:
    def __init__(self, brain, settings: AutoSettings | None=None):
        self.brain=brain
        self.settings=settings or AutoSettings.from_env()
        self._stop=threading.Event()
        self._thread=None
        self._ticks=0
        self._last_growth_attempt_steps=-100000
        self._pending_research=False

    def start(self):
        if not self._thread:
            self._thread=threading.Thread(target=self._loop,name='nord-brain-learning',daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join()

    def _loop(self):
        while not self._stop.wait(self.settings.interval):
            try:self.tick()
            except Exception as e:
                with self.brain.lock:
                    self.brain.memory.event('auto_error',repr(e)[:160])

    def tick(self):
        self._ticks+=1
        settings=self.settings
        brain=self.brain
        # Do not block the interactive terminal on an Internet request.
        if settings.research and self._ticks%3==0:
            with brain.lock:
                topic=brain.memory.next_unknown()
            if topic:
                try:
                    results=search(topic)
                    with brain.lock:
                        n=brain.absorb_official_sources(topic,results)
                        brain.memory.unknown_searched(topic)
                        brain.memory.event('research',f'{topic[:75]}; source_count={len(results)}; trained={n}',float(n))
                except Exception as e:
                    with brain.lock:
                        brain.memory.unknown_searched(topic)
                        brain.memory.event('research_error',repr(e)[:160])
        if not settings.training:
            return
        with brain.lock:
            # Hypothesis lab follows the teacher's 2+2=4 seed: generate novel
            # equivalent/contrast equations, predict, VERIFY, then train.
            # Source questions and held-out math test questions are excluded.
            for question,expected in brain.memory.pending_math_hypotheses(settings.hypotheses_per_tick):
                if brain.memory.is_holdout(question):
                    brain.memory.mark_math_hypothesis(question,'heldout','rejected')
                    continue
                proof=math_question(question)
                checked=(proof.steps[-1].split(' = ')[-1] if proof and proof.steps else None)
                if checked!=expected:
                    brain.memory.mark_math_hypothesis(question,'invalid target','rejected')
                    brain.memory.event('hypothesis_rejected',question)
                    continue
                before=brain.core.generate(question,max_bytes=8,temperature=0).strip()
                brain._learn(question,expected,epochs=1,origin='verified_math_hypothesis')
                brain.memory.mark_math_hypothesis(question,before,'trained')
                brain.memory.metric('hypothesis_pretrain_correct',float(before==expected))
                brain.memory.event('hypothesis_trained',f'{question} = {expected}; predicted={before[:12]}')
            if settings.world_simulations:
                result=brain.world.experience_cycle(scenarios=settings.world_scenarios,
                                                    steps_per_world=12,updates=16)
                brain.memory.event('world_training',
                    f"scenarios={result['scenarios']}; observations={result['observations']}",
                    result['after']['joint_loss'])
            if settings.experiments:
                level=max(0,min(MAX_LEVEL,int(brain.memory.get_meta('math_level','0'))))
                training, heldout=curriculum(level)
                # Evaluation tasks are frozen, untrained, and excluded from replay.
                available=[]
                for question,answer in heldout:
                    if brain.memory.is_holdout(question):
                        available.append((question,answer))
                    else:
                        try:
                            brain.memory.holdout(question,answer)
                            available.append((question,answer))
                        except ValueError:
                            # Already taught in an older release: not an unseen test.
                            pass
                cursor=int(brain.memory.get_meta('curriculum_cursor','0'))
                completed=[]
                # Train on *different* procedurally generated examples each cycle.
                # Validate targets mathematically; never use the model's own guess
                # as the gold answer. Save once after the group for CPU efficiency.
                brain.core.save()
                try:
                    for _ in range(settings.examples_per_tick):
                        question,answer=training[cursor%len(training)]
                        cursor+=1
                        if brain.memory.is_holdout(question):
                            continue
                        loss=brain.core.train(question,answer,epochs=1)
                        completed.append((question,answer,loss))
                    brain.core.save()
                except Exception:
                    brain.core.load()
                    raise
                for question,answer,loss in completed:
                    brain.memory.teach(question,answer,origin='verified_experiment')
                    brain.memory.metric('train_loss',loss)
                brain.memory.set_meta('curriculum_cursor',str(cursor))
                brain.memory.event('math_training',f'level={level}; examples={len(completed)}')
                # Independent, neural-only measurement. The symbolic verifier
                # is NOT consulted for these predictions or counted as success.
                if self._ticks%4==0 and len(available)>=6:
                    probes=available[:8]
                    correct=0
                    for question,answer in probes:
                        prediction=brain.core.generate(question,max_bytes=8,temperature=0).strip()
                        correct+=prediction==answer
                    accuracy=correct/len(probes)
                    brain.memory.set_meta('math_neural_accuracy',f'{correct}/{len(probes)}')
                    brain.memory.event('math_neural_eval',f'level={level}; neural_only',accuracy)
                    # Promote only on actual unseen success, never from training loss.
                    if accuracy>=.75 and level<MAX_LEVEL:
                        brain.memory.set_meta('math_level',str(level+1))
                        brain.memory.set_meta('curriculum_cursor','0')
                        brain.memory.event('math_level_up',f'{level} -> {level+1}',accuracy)
                # Practice compositional consequences of teacher-provided facts.
                if self._ticks%2==0:
                    example=brain.reasoner.derived_example(self._ticks//2)
                    if example and not brain.memory.is_holdout(example[0]):
                        try:
                            brain._learn(*example,epochs=1,origin='reasoning_proof')
                            brain.memory.event('reasoning_practice','teacher-grounded deduction')
                        except ValueError as e:
                            brain.memory.event('reasoning_skipped',str(e)[:140])
            # Replay is also learning: old examples are reintroduced without an
            # extra teacher command. Keep CPU use small on 1-vCPU machines.
            recent=brain.memory.lessons(8)
            if recent:
                q,a=recent[self._ticks%len(recent)]
                brain.core.save()
                try:
                    loss=brain.core.train(q,a,epochs=1)
                    brain.core.save()
                except Exception:
                    brain.core.load()
                    raise
                brain.memory.metric('auto_replay_loss',loss)
            # Evaluate no more than every fourth autonomous cycle.
            if self._ticks%4==0:
                tests=brain.memory.tests(12)
                if len(tests)>=6:
                    score=brain.core.score(tests)
                    brain.memory.event('eval_loss','independent_holdouts',score)
                    if settings.growth:
                        self._maybe_grow(tests)

    def _maybe_grow(self,tests):
        b=self.brain
        p=self.settings
        scores=b.memory.recent_losses(3)
        if b.core.steps-self._last_growth_attempt_steps<80:return
        if not growth_candidate(scores,b.core.steps,b.memory.unique_lesson_count(),len(tests),
                                b.core.parameter_count,b.core.hidden_size,p.max_params,p.max_width):return
        self._last_growth_attempt_steps=b.core.steps
        initial=b.core.score(tests)
        replay=b.memory.lessons(8)
        if not replay:return
        old_size=b.core.hidden_size
        # Compare equal training with and without structural growth. Otherwise,
        # an apparent gain could merely be caused by extra gradient steps.
        b.core.save()
        try:
            for q,a in reversed(replay):
                b.core.train(q,a,epochs=1)
            control_score=b.core.score(tests)
            b.core.load()
            b.core.grow(16)
            if b.core.parameter_count>p.max_params:
                b.core.load()
                b.memory.event('growth_rejected','hardware budget')
                return
            for q,a in reversed(replay):
                b.core.train(q,a,epochs=1)
            candidate_score=b.core.score(tests)
            # The change must outperform the unexpanded control on unseen
            # examples, not simply fit the same examples harder.
            if (math.isfinite(candidate_score) and candidate_score<initial-0.01
                    and candidate_score<control_score-0.01):
                b.core.save()
                b.memory.event('growth_accepted',f'{old_size} -> {b.core.hidden_size}',control_score-candidate_score)
            else:
                b.core.load()
                b.memory.event('growth_rejected','no benefit versus control',candidate_score-control_score)
        except Exception:
            b.core.load()
            raise
