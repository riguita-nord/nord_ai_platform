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

from .experiments import curriculum
from .internet import search


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
    interval: int = 60
    training: bool = True
    experiments: bool = True
    growth: bool = True
    research: bool = False
    max_params: int = 3000000
    max_width: int = 1024

    @classmethod
    def from_env(cls):
        return cls(interval=integer('NORD_AUTO_INTERVAL',60,5,3600),
                   training=on('NORD_AUTO_TRAIN',True),
                   experiments=on('NORD_AUTO_EXPERIMENTS',True),
                   growth=on('NORD_AUTO_GROW',True),
                   research=on('NORD_AUTONOMOUS_RESEARCH',True) and on('NORD_INTERNET'),
                   max_params=integer('NORD_MAX_PARAMS',3000000,100000,10000000),
                   max_width=integer('NORD_MAX_MLP_WIDTH',1024,128,4096))


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
                        for item in results:
                            brain.memory.observation(topic,item['title'],item['url'],item['summary'])
                        brain.memory.unknown_searched(topic)
                        brain.memory.event('research',topic,float(len(results)))
                except Exception as e:
                    with brain.lock:
                        brain.memory.unknown_searched(topic)
                        brain.memory.event('research_error',repr(e)[:160])
        if not settings.training:
            return
        with brain.lock:
            if settings.experiments:
                training, tests=curriculum()
                # Only add independent test items that were not already taught.
                for question,answer in tests:
                    if not brain.memory.is_holdout(question):
                        try:brain.memory.holdout(question,answer)
                        except ValueError:pass
                cursor=int(brain.memory.get_meta('curriculum_cursor','0'))
                question,answer=training[cursor%len(training)]
                if not brain.memory.is_holdout(question):
                    brain._learn(question,answer,epochs=1,origin='verified_experiment')
                brain.memory.set_meta('curriculum_cursor',str(cursor+1))
                # Practice compositional consequences of teacher-provided facts.
                # Every label is derived by a bounded proof, not invented by the model.
                if self._ticks%2==0:
                    example=brain.reasoner.derived_example(self._ticks//2)
                    if example and not brain.memory.is_holdout(example[0]):
                        try:
                            brain._learn(*example,epochs=1,origin='reasoning_proof')
                            brain.memory.event('reasoning_practice','teacher-grounded deduction')
                        except ValueError as e:
                            # A derived sentence can exceed tiny context/training limits.
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
