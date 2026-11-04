"""Cross-domain hypothesis laboratory with bounded CPU-only weight updates.

The same experimental protocol is applied to *supported* domains:
  propose from a trusted source -> predict with neural weights -> verify using
  independent evidence -> update weights -> evaluate held-out probes.

This does NOT make the language model generally intelligent. Ground-truth
math/pattern rules, teacher assertions and a virtual world are different types
of evidence, and their source must remain visible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .experiments import curriculum, MAX_LEVEL
from .reasoning import math_question

if TYPE_CHECKING:
    from .runtime import Brain

DOMAINS = ('matematica', 'padroes', 'logica', 'linguagem', 'mundo')


@dataclass(frozen=True)
class Case:
    domain: str
    question: str
    expected: str
    evidence: str


def pattern_case(index: int) -> Case:
    """Generate a novel arithmetic progression. A hidden arithmetic rule judges it.

    Indexes congruent to 0 (mod 7) are *never* trained by this laboratory.
    This exclusion is independent of the model's current output.
    """
    if index < 0:
        raise ValueError('Índice não pode ser negativo')
    start = (index * 11 + (index // 7) * 3) % 27
    step = 1 + (index * 7 + index // 9) % 9
    n = 3 + (index // 5) % 2
    seq = [start + step * i for i in range(n)]
    q = 'Completa a sequência: ' + ', '.join(map(str, seq)) + ', ?'
    return Case('padroes', q, str(start + step * n), 'progressão aritmética verificada')


def verify_pattern(question: str, expected: str) -> bool:
    """Derive the next value from the progression, never from the model reply."""
    import re
    m = re.fullmatch(r'Completa a sequência: (\d+(?:, \d+){2,3}), \?', question)
    if not m:
        return False
    values = [int(v) for v in m.group(1).split(', ')]
    steps = [y - x for x,y in zip(values,values[1:])]
    return bool(steps) and all(x == steps[0] for x in steps) and expected == str(values[-1]+steps[0])


class CognitiveLab:
    """Runs small real experiments and records origins and outcomes per domain."""
    def __init__(self, brain: 'Brain'):
        self.brain = brain

    def _math(self) -> Case | None:
        b = self.brain
        level = max(0,min(MAX_LEVEL,int(b.memory.get_meta('math_level','0'))))
        training, tests = curriculum(level)
        # Legacy math holdouts are added by the existing autonomous curriculum.
        # Ensure they are registered even for on-demand laboratory sessions.
        for q,a in tests:
            if not b.memory.is_holdout(q):
                try:
                    b.memory.holdout(q,a)
                except ValueError:
                    pass  # old teacher data cannot retrospectively become a holdout
        cursor = int(b.memory.get_meta('cognitive_math_cursor','0'))
        for i in range(min(12,len(training))):
            q,a = training[(cursor+i)%len(training)]
            if not b.memory.is_holdout(q) and not b.memory.is_cognitive_probe(q):
                b.memory.set_meta('cognitive_math_cursor',str(cursor+i+1))
                return Case('matematica',q,a,'aritmética verificada por cálculo independente')
        b.memory.set_meta('cognitive_math_cursor',str(cursor+12))
        return None

    def _patterns(self) -> Case | None:
        b=self.brain
        cursor=int(b.memory.get_meta('cognitive_pattern_cursor','1'))
        # Reserve fixed, non-trained probes. The separate domain-specific
        # reservation prevents them leaking through cognitive training/replay.
        for i in range(0,70,7):
            candidate=pattern_case(i)
            b.memory.reserve_cognitive_probe('padroes',candidate.question,candidate.expected)
        for i in range(cursor,cursor+20):
            case=pattern_case(i)
            if i % 7 != 0 and not b.memory.is_cognitive_probe(case.question):
                b.memory.set_meta('cognitive_pattern_cursor',str(i+1))
                return case
        b.memory.set_meta('cognitive_pattern_cursor',str(cursor+20))
        return None

    def _language(self) -> Case | None:
        b=self.brain
        concepts=b.memory.teacher_concepts(limit=40)
        if not concepts:
            return None
        cursor=int(b.memory.get_meta('cognitive_language_cursor','0'))
        for i in range(len(concepts)):
            term,definition=concepts[(cursor+i)%len(concepts)]
            # Only teacher-supplied descriptions are used, never scraped text.
            if len(term)>45 or len(definition)>200 or not definition.strip():
                continue
            probe=f'Explica o significado de {term}.'
            b.memory.reserve_cognitive_probe('linguagem',probe,definition)
            q=f'Define o conceito {term}.' if (cursor+i)%2==0 else f'Fala-me sobre {term}.'
            if not b.memory.is_cognitive_probe(q):
                b.memory.set_meta('cognitive_language_cursor',str(cursor+i+1))
                return Case('linguagem',q,definition,'definição fornecida pelo utilizador (não verificada externamente)')
        return None

    def _logic(self) -> Case | None:
        b=self.brain
        if b.memory.reasoning_fact_count()<2:
            return None
        cursor=int(b.memory.get_meta('cognitive_logic_cursor','0'))
        for i in range(cursor,cursor+12):
            derived=b.reasoner.derived_example(i)
            if derived is None:
                return None
            question,_=derived
            proof=b.reasoner.solve(question)
            if proof is None or proof.method!='graph':
                continue
            # Multiple transformations can map onto a single inferred question.
            # The alternative question form is reserved for neural-only tests.
            probe=question.removesuffix('?')+'?'
            # Probes are distinct from training questions, with the same fact
            # semantics but different phrasing.
            if ' é ' in probe:
                probe='Será que '+probe[0].lower()+probe[1:]
            elif ' tem ' in probe:
                probe='Será que '+probe[0].lower()+probe[1:]
            if probe!=question:
                b.memory.reserve_cognitive_probe('logica',probe,proof.answer)
            if not b.memory.is_cognitive_probe(question):
                b.memory.set_meta('cognitive_logic_cursor',str(i+1))
                return Case('logica',question,proof.answer,'dedução de premissas ensinadas pelo utilizador')
        return None

    def candidate(self, domain: str) -> Case | None:
        methods={'matematica':self._math,'padroes':self._patterns,
                 'linguagem':self._language,'logica':self._logic}
        if domain not in methods:
            raise ValueError('Área desconhecida para experiências textuais')
        return methods[domain]()

    def verify(self, case: Case) -> bool:
        b=self.brain
        if case.domain=='matematica':
            proof=math_question(case.question)
            return bool(proof and proof.steps and proof.steps[-1].split(' = ')[-1]==case.expected)
        if case.domain=='padroes':
            return verify_pattern(case.question,case.expected)
        if case.domain=='logica':
            result=b.reasoner.solve(case.question)
            return result is not None and result.method=='graph' and result.answer==case.expected
        if case.domain=='linguagem':
            import re
            m=re.fullmatch(r'(?:Define o conceito|Fala-me sobre) (.+)\.',case.question)
            return bool(m and b.memory.lookup_concept(m[1])==case.expected
                        and b.memory.teacher_has_concept(m[1]))
        return False

    def trial(self,domain: str) -> dict:
        """One prediction, independent verification, optional real weight update.

        All actions are inside the caller's Brain lock. The output is suitable
        for an honest terminal report; it does not fake neural competence.
        """
        b=self.brain
        case=self.candidate(domain)
        if case is None:
            return {'domain':domain,'status':'sem exemplos verificados disponíveis'}
        if b.memory.is_holdout(case.question) or b.memory.is_cognitive_probe(case.question):
            return {'domain':domain,'status':'pergunta reservada para avaliação; não treinada'}
        if not self.verify(case):
            return {'domain':domain,'status':'hipótese rejeitada: não foi possível verificar'}
        token_budget={'matematica':8,'padroes':8,'logica':20,'linguagem':32}[domain]
        predicted=b.core.generate(case.question,max_bytes=token_budget,temperature=0).strip()
        predicted_correct=predicted.casefold()==case.expected.strip().casefold()
        try:
            loss=b._learn(case.question,case.expected,epochs=1,origin='cognitive_'+domain)
        except ValueError as exc:
            return {'domain':domain,'status':'treino não realizado: '+str(exc)[:100]}
        b.memory.cognitive_trial(domain,case.question,case.expected,predicted,
                                 case.evidence,float(loss),predicted_correct)
        return {'domain':domain,'status':'treinado','question':case.question,
                'prediction':predicted or '(sem resposta)',
                'verified':case.expected,'neural_correct_before':predicted_correct,
                'loss':round(float(loss),4),'evidence':case.evidence}

    def evaluate(self,domain: str, limit: int = 2) -> dict | None:
        """Raw neural-only accuracy on frozen, never-trained questions."""
        b=self.brain
        limit=max(1,min(limit,6))
        if domain=='matematica':
            level=max(0,min(MAX_LEVEL,int(b.memory.get_meta('math_level','0'))))
            _,test=curriculum(level)
            tests=[(q,a) for q,a in test if b.memory.is_holdout(q)]
        else:
            tests=b.memory.cognitive_probes(domain,limit=60)
        # A legacy model might have trained on these questions before this
        # module was introduced. Do not count those as independent holdouts.
        tests=[(q,a) for q,a in tests if not b.memory.lesson_was_trained(q)]
        if not tests:
            return None
        correct=0
        for q,a in tests[:limit]:
            max_bytes=8 if domain in ('matematica','padroes') else 40
            pred=b.core.generate(q,max_bytes=max_bytes,temperature=0).strip()
            correct+=pred.casefold()==a.strip().casefold()
        total=min(limit,len(tests))
        accuracy=correct/total
        b.memory.event('cognitive_evaluation',f'{domain}: {correct}/{total} apenas pesos',accuracy)
        return {'correct':correct,'total':total,'accuracy':round(accuracy,4)}

    def run(self,domain: str='tudo', *, examples: int=1, evaluate: bool=True) -> dict:
        """Execute up to a small bounded number of experiments per area.

        The world domain uses its separate sensorimotor model and returns
        continuous held-out prediction error, not textual answer accuracy.
        """
        b=self.brain
        if domain not in (*DOMAINS,'tudo'):
            raise ValueError('Usa tudo, matematica, padroes, logica, linguagem ou mundo')
        if not 1<=examples<=5:
            raise ValueError('Entre 1 e 5 experiências por área')
        chosen=DOMAINS if domain=='tudo' else (domain,)
        result={'domains':{},'neural_training_steps_before':b.core.steps}
        for area in chosen:
            if area=='mundo':
                world=b.world.experience_cycle(scenarios=examples,steps_per_world=12,
                                               updates=12,save=True)
                b.memory.event('cognitive_world',f'{examples} mundos virtuais',world['after']['joint_loss'])
                result['domains'][area]={'status':'treinado','scenarios':examples,
                    'prediction_error_before':world['before']['joint_loss'],
                    'prediction_error_after':world['after']['joint_loss'],
                    'goals_reached':world['goals_reached'],'collisions':world['collisions']}
                continue
            trials=[self.trial(area) for _ in range(examples)]
            status={'trials':trials}
            if evaluate and any(t.get('status')=='treinado' for t in trials):
                status['heldout']=self.evaluate(area,limit=2)
            result['domains'][area]=status
        result['neural_training_steps_after']=b.core.steps
        return result

    def describe(self,report: dict) -> str:
        names={'matematica':'Matemática','padroes':'Padrões e raciocínio mental',
               'logica':'Lógica e relações','linguagem':'Linguagem e conceitos',
               'mundo':'Mundo virtual e decisões'}
        lines=['Laboratório da Nord Brain — experiências e aprendizagem em CPU:']
        for key,info in report['domains'].items():
            lines.append('\n'+names[key]+':')
            if key=='mundo':
                lines.append(f"  • {info['scenarios']} cenários; {info['goals_reached']} objetivos alcançados; {info['collisions']} colisões tentadas")
                lines.append(f"  • Erro de previsão em mundos separados: {info['prediction_error_before']:.4f} → {info['prediction_error_after']:.4f}")
                continue
            for t in info['trials']:
                if t['status']!='treinado':
                    lines.append('  • '+t['status'])
                else:
                    lines.append(f"  • Teste: {t['question']} | previsão neuronal: {t['prediction']} | confirmado: {t['verified']}")
                    lines.append('    '+('Já acertava antes do treino.' if t['neural_correct_before'] else 'Falhou antes do treino; pesos atualizados com resposta verificada.'))
            held=info.get('heldout')
            if held:
                lines.append(f"  • Avaliação neuronal em perguntas reservadas: {held['correct']}/{held['total']} (não vista no treino)")
        lines.append('\nEstas experiências medem treino nos pesos; os verificadores e os conceitos ensinados não constituem inteligência geral.')
        return '\n'.join(lines)
