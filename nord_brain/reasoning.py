"""Grounded, limited compositional reasoning for the terminal-first Nord Brain.

This is an explicit *symbolic verifier* working alongside the trainable neural
model, not proof that the randomly-initialised Transformer reasons on its own.
Teacher propositions are premises, not automatically verified real-world facts.
No LLM self-generated labels or untrusted web text are imported into the graph.
"""
from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import re


MAX_EDGES = 300
MAX_DEPTH = 8


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str  # is_a | has
    object: str
    statement: str


@dataclass(frozen=True)
class Reasoning:
    answer: str
    steps: tuple[str, ...]
    method: str  # arithmetic | graph | incomplete


def tidy(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().strip('“”"')).rstrip('.?! ')


def singular(token: str) -> str:
    irregular = {'animais':'animal','pessoais':'pessoal','cães':'cão','pães':'pão','mães':'mãe',
                 'leões':'leão','informações':'informação','ações':'ação'}
    if token in irregular:
        return irregular[token]
    if token.endswith('ões') and len(token) > 5:
        return token[:-3] + 'ão'
    if token.endswith('ais') and len(token) > 5:
        return token[:-3] + 'al'
    if token.endswith('res') and len(token) > 5:
        return token[:-1]
    if token.endswith('s') and len(token) > 4 and not token.endswith(('ss','us','is')):
        return token[:-1]
    return token


def canonical(phrase: str, plural: bool = False) -> str:
    phrase = tidy(phrase).casefold()
    phrase = re.sub(r'^(?:o|a|os|as|um|uma|uns|umas)\s+', '', phrase)
    if plural:
        parts = phrase.split()
        if parts:
            parts[0] = singular(parts[0])
            phrase = ' '.join(parts)
    return phrase


def _safe_component(s: str) -> bool:
    return 1 <= len(s) <= 70 and bool(re.fullmatch(r'[\wÀ-ÿ-]+(?:[ \wÀ-ÿ-]*[\wÀ-ÿ-])?', s, re.UNICODE))


def parse_fact(message: str) -> Fact | None:
    raw = tidy(message)
    if not raw or '?' in raw or '=>' in raw:
        return None
    # Explicit universal rules (only the teacher can make universal premises).
    universal = re.fullmatch(r'(?:todos os|todas as|todo|toda)\s+(.+?)\s+(são|é|têm|tem)\s+(.+)', raw, re.I)
    if universal:
        lhs,verb,rhs = universal.groups()
        predicate = 'is_a' if verb.casefold() in ('são','é') else 'has'
        subject, obj = canonical(lhs,plural=True), canonical(rhs,plural=predicate=='is_a' and verb.casefold()=='são')
        if _safe_component(subject) and _safe_component(obj) and subject != obj:
            return Fact(subject,predicate,obj,raw)
        return None
    # Concrete or generic taxonomic/attribute statements.
    match = re.fullmatch(r'(.+?)\s+(é|tem)\s+(.+)', raw, re.I)
    if not match:
        return None
    lhs,verb,rhs = match.groups()
    # Do not treat questions, negations or modal uncertainties as affirmative facts.
    if re.search(r'\b(?:não|talvez|possivelmente|acho|penso|pode|parece)\b',raw,re.I):
        return None
    subject = canonical(lhs)
    predicate = 'is_a' if verb.casefold() == 'é' else 'has'
    obj = canonical(rhs)
    if _safe_component(subject) and _safe_component(obj) and subject != obj:
        return Fact(subject,predicate,obj,raw)
    return None


def _render(fr: Fraction) -> str:
    if fr.denominator == 1:
        return str(fr.numerator)
    return f'{fr.numerator}/{fr.denominator}'


def calculate(expr: str) -> Reasoning | None:
    expr = expr.strip().replace('×','*').replace('÷','/').replace('−','-').replace('x','*')
    if len(expr)>75 or not re.fullmatch(r'[0-9+*/().\s-]+',expr):
        return None
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return None
    steps: list[str] = []
    symbols = {ast.Add:'+',ast.Sub:'-',ast.Mult:'×',ast.Div:'/'}
    def walk(node,depth=0):
        if depth>12:raise ValueError('expressão demasiado profunda')
        if isinstance(node, ast.Expression):
            return walk(node.body,depth+1)
        if isinstance(node,ast.Constant) and type(node.value) is int:
            if abs(node.value)>1_000_000:raise ValueError('valor acima do limite')
            return Fraction(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            v=walk(node.operand,depth+1)
            return v if isinstance(node.op,ast.UAdd) else -v
        if isinstance(node,ast.BinOp) and type(node.op) in symbols:
            a=walk(node.left,depth+1)
            b=walk(node.right,depth+1)
            if isinstance(node.op,ast.Add):v=a+b
            elif isinstance(node.op,ast.Sub):v=a-b
            elif isinstance(node.op,ast.Mult):v=a*b
            else:
                if not b:raise ValueError('divisão por zero')
                v=a/b
            if abs(v)>1_000_000_000 or v.denominator>1_000_000_000:
                raise ValueError('resultado acima do limite')
            steps.append(f'{_render(a)} {symbols[type(node.op)]} {_render(b)} = {_render(v)}')
            return v
        raise ValueError('operação não permitida')
    try:
        val=walk(tree)
    except (ValueError,ZeroDivisionError,OverflowError):
        return None
    if not steps:return None
    return Reasoning(f'O resultado é {_render(val)}. '+('Passos: '+'; '.join(steps) if len(steps)>1 else steps[0]),tuple(steps),'arithmetic')


def math_question(s: str) -> Reasoning | None:
    s=tidy(s)
    m=re.match(r'^(?:quanto é|calcula|calcular|resolve|qual é o resultado de)\s+(.+)$',s,re.I)
    if not m:return None
    return calculate(m[1])


def story_problem(message: str) -> Reasoning | None:
    # Restricted template with explicit quantities; no guesses about ambiguous stories.
    s=tidy(message).casefold()
    m=re.fullmatch(r'se (?:eu )?(?:tenho|tinha)\s+(\d{1,5})\s+([\wà-ÿ]+)\s+e\s+(?:eu )?(ganho|recebo|compro|dou|perco|gasto)\s+(\d{1,5})\s+(?:\2)?\s*,?\s*quant[oa]s?\s+(?:\2\s+)?(?:tenho|fico|restam|me restam)(?:\s+agora)?',s,re.I)
    if not m:return None
    initial,object_name,action,delta=m.groups()
    a,b=int(initial),int(delta)
    sign=1 if action in ('ganho','recebo','compro') else -1
    answer=a+sign*b
    steps=(f'Quantidade inicial: {a} {object_name}',f'{action}: {b} {object_name}',f'{a} {"+" if sign>0 else "-"} {b} = {answer}')
    return Reasoning(f'Ficas com {answer} {object_name}. '+ '; '.join(steps),steps,'arithmetic')


class ReasoningEngine:
    def __init__(self,memory):
        self.memory=memory

    def learn(self,message: str) -> Fact | None:
        fact=parse_fact(message)
        if fact:
            self.memory.add_reasoning_fact(fact.subject,fact.predicate,fact.object,fact.statement)
        return fact

    def facts(self):
        return self.memory.reasoning_facts(MAX_EDGES)

    def _graph(self, extra: tuple[Fact, ...] = ()):
        edges={}
        properties={}
        saved=self.facts()
        for subject,pred,obj,statement in saved + [(f.subject,f.predicate,f.object,f.statement) for f in extra]:
            if pred=='is_a':edges.setdefault(subject,[]).append((obj,statement))
            elif pred=='has':properties.setdefault(subject,[]).append((obj,statement))
        return edges,properties

    @staticmethod
    def _path(start,target,edges):
        queue=deque([(start,())])
        seen={start}
        while queue:
            node,path=queue.popleft()
            if node==target and path:return path
            if len(path)>=MAX_DEPTH:continue
            for nxt,statement in edges.get(node,[]):
                if nxt not in seen:
                    new=path+(statement,)
                    if nxt==target:return new
                    seen.add(nxt)
                    queue.append((nxt,new))
        return None

    @staticmethod
    def _ancestors(start,edges):
        queue=deque([(start,())]);seen={start}
        while queue:
            node,path=queue.popleft()
            yield node,path
            if len(path)>=MAX_DEPTH:continue
            for nxt,statement in edges.get(node,[]):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt,path+(statement,)))

    def _solve_relation(self,subject,predicate,obj,why=False,extra: tuple[Fact, ...] = ()):
        edges,props=self._graph(extra)
        subject,obj=canonical(subject),canonical(obj)
        path=None
        if predicate=='is_a':
            path=self._path(subject,obj,edges)
        elif predicate=='has':
            for node,pre_path in self._ancestors(subject,edges):
                for attribute,statement in props.get(node,[]):
                    if attribute==obj:
                        path=pre_path+(statement,)
                        break
                if path:break
        if path:
            return Reasoning('Sim. '+('Porque: ' if why else 'Dedução: ')+ ' → '.join(path)+'.',path,'graph')
        return Reasoning('Ainda não consigo concluir isso a partir das relações que me ensinaste. Podes explicar-me a ligação?',(), 'incomplete')

    def _describe(self,subject):
        edges,props=self._graph()
        subject=canonical(subject)
        details=[]
        for node,path in self._ancestors(subject,edges):
            if path:details.append(' → '.join(path))
            for _,statement in props.get(node,[]):
                details.append(' → '.join(path+(statement,)))
        if not details:return None
        unique=list(dict.fromkeys(details))[:6]
        return Reasoning('Sei estas relações que me ensinaste: ' + '; '.join(unique)+'.',tuple(unique),'graph')

    def solve(self,message: str) -> Reasoning | None:
        s=tidy(message)
        numeric=math_question(s) or story_problem(s)
        if numeric:return numeric
        # Temporary hypothetical premises are used only for this answer.
        # They are NEVER written to the teacher's permanent knowledge graph.
        hypo=re.fullmatch(r'se\s+(.+?)\s+e\s+(.+?)[,;]\s*(?:então\s+)?(.+)',s,re.I)
        if hypo:
            premises=tuple(parse_fact(x) for x in hypo.groups()[:2])
            conclusion=hypo.group(3)
            q=re.fullmatch(r'(.+?)\s+(é|tem)\s+(.+)',conclusion,re.I)
            if all(premises) and q:
                sub,verb,obj=q.groups()
                result=self._solve_relation(sub,'is_a' if verb.casefold()=='é' else 'has',obj,extra=premises)
                return Reasoning('Assumindo as duas hipóteses: '+result.answer,result.steps,result.method)
        if re.match(r'^(?:quanto é|calcula|calcular|resolve|qual é o resultado de)\b',s,re.I):
            return Reasoning('Não consegui verificar essa expressão matemática. Podes reformulá-la?',(), 'incomplete')
        if not (message.strip().endswith('?') or re.match(r'^(?:explica|porque|por que|o que sabes)\b',s,re.I)):
            return None
        m=re.fullmatch(r'(?:o que sabes sobre|o que é)\s+(.+)',s,re.I)
        if m:return self._describe(m[1])
        m=re.fullmatch(r'(?:por ?que|porque)(?: é que)?\s+(.+?)\s+(é|tem)\s+(.+)',s,re.I)
        if m:
            subj,verb,obj=m.groups()
            return self._solve_relation(subj,'is_a' if verb.casefold()=='é' else 'has',obj,why=True)
        m=re.fullmatch(r'(.+?)\s+(é|tem)\s+(.+)',s,re.I)
        if m:
            subj,verb,obj=m.groups()
            return self._solve_relation(subj,'is_a' if verb.casefold()=='é' else 'has',obj)
        return None

    def derived_example(self,index: int):
        """Return an inferred example grounded in teacher facts for neural replay.

        This is a training demonstration. It does not count as neural proof.
        No web information is used as teacher truth.
        """
        edges,props=self._graph()
        candidates=[]
        for start in sorted(edges):
            for node,path in self._ancestors(start,edges):
                if len(path)>=2:
                    candidates.append((f'{start} é {node}?', 'Sim. '+' → '.join(path)+'.'))
                if path:
                    for attribute,statement in props.get(node,[]):
                        candidates.append((f'{start} tem {attribute}?','Sim. '+' → '.join(path+(statement,))+'.'))
        if not candidates:return None
        return candidates[index%len(candidates)]
