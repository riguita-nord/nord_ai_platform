"""Checks inference from teacher premises, no hallucinated logical negatives,
and verifies arithmetic evaluation does not execute arbitrary Python.
"""
import tempfile
import unittest
from pathlib import Path
from nord_brain.storage import Memory
from nord_brain.reasoning import ReasoningEngine, calculate, parse_fact, story_problem
from nord_brain.runtime import Brain
from nord_brain.core import NeuralCore, TransformerConfig

CONFIG=TransformerConfig(d_model=32,layers=2,heads=4,kv_heads=2,intermediate_size=64,context=128)


class ReasoningUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.memory=Memory(Path(self.temp.name)/'mem.sqlite3')
        self.engine=ReasoningEngine(self.memory)

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def teach(self,*statements):
        for x in statements:self.assertIsNotNone(self.engine.learn(x),x)

    def test_three_step_taxonomy(self):
        self.teach('A Luna é um gato.', 'Todo gato é um mamífero.', 'Todos os mamíferos são animais.')
        result=self.engine.solve('A Luna é um animal?')
        self.assertEqual(result.method,'graph')
        self.assertEqual(len(result.steps),3)
        self.assertIn('Sim.',result.answer)
        desc=self.engine.solve('O que é Luna?')
        self.assertIn('mamífero',desc.answer)

    def test_property_inheritance(self):
        self.teach('A Lua é um gato.', 'Todos os gatos são mamíferos.', 'Todos os mamíferos têm coração.')
        result=self.engine.solve('A Lua tem coração?')
        self.assertIn('Sim',result.answer)
        self.assertEqual(len(result.steps),3)
        explanation=self.engine.solve('Porque é que a Lua tem coração?')
        self.assertIn('Porque:',explanation.answer)

    def test_unknown_is_not_false_and_no_cycles(self):
        self.teach('A é B.', 'B é A.')
        result=self.engine.solve('A é C?')
        self.assertEqual(result.method,'incomplete')
        self.assertNotIn('Não.',result.answer)

    def test_teacher_only_and_no_web_label(self):
        self.assertIsNone(parse_fact('Talvez um gato é um animal.'))
        self.assertIsNone(parse_fact('Um gato não é uma planta.'))
        with self.assertRaises(ValueError):
            self.memory.add_reasoning_fact('x','is_a','y','web says',origin='web')
        self.assertEqual(self.memory.reasoning_fact_count(),0)

    def test_arithmetic_precedence_and_explanations(self):
        result=calculate('2 + 3 * 4')
        self.assertIn('14',result.answer)
        self.assertEqual(len(result.steps),2)
        self.assertIn('20',calculate('(2 + 3) * 4').answer)
        self.assertIn('3',calculate('9 / 3').answer)
        self.assertIn('3/2',calculate('3 / 2').answer)

    def test_arithmetic_rejects_execution_and_unbounded(self):
        self.assertIsNone(calculate('__import__("os").system("echo hacked")'))
        self.assertIsNone(calculate('1 / 0'))
        self.assertIsNone(calculate('2**1000'))
        self.assertIsNone(calculate('99999999999999 + 1'))

    def test_curriculum_is_disjoint_and_verified(self):
        from nord_brain.experiments import curriculum
        from nord_brain.reasoning import math_question
        train,holdout=curriculum()
        self.assertTrue(set(q for q,a in train).isdisjoint(q for q,a in holdout))
        self.assertGreater(len(holdout),10)
        sample=train[::20]+holdout[::5]
        for question,answer in sample:
            result=math_question(question)
            self.assertIsNotNone(result)
            self.assertTrue(result.answer.startswith(f'O resultado é {answer}.'),question)

    def test_story_problem(self):
        result=story_problem('Se tenho 5 maçãs e recebo 3 maçãs, quantas maçãs tenho?')
        self.assertIn('8 maçãs',result.answer)

    def test_temporary_hypotheses_do_not_pollute_memory(self):
        question='Se Luna é um gato e todo gato é mamífero, Luna é mamífero?'
        r=self.engine.solve(question)
        self.assertIn('Sim.',r.answer)
        self.assertEqual(len(r.steps),2)
        self.assertEqual(self.memory.reasoning_fact_count(),0)
        self.assertEqual(self.engine.solve('Luna é mamífero?').method,'incomplete')

    def test_persisted_deductions_and_derived_example(self):
        self.teach('Luna é um gato.', 'Todo gato é um mamífero.', 'Todos os mamíferos são animais.')
        example=self.engine.derived_example(0)
        self.assertIsNotNone(example)
        self.assertIn('Sim.',example[1])
        again=ReasoningEngine(Memory(Path(self.temp.name)/'mem.sqlite3'))
        try:
            self.assertIn('Sim.',again.solve('Luna é um animal?').answer)
        finally:again.memory.close()


class RuntimeReasoningTests(unittest.TestCase):
    def test_autonomous_practice_from_proven_relations(self):
        from nord_brain.autonomy import Autonomy, AutoSettings
        with tempfile.TemporaryDirectory() as temp:
            brain=Brain(Path(temp),start_autonomy=False)
            brain.core=NeuralCore(Path(temp)/'brain-v2.pt',CONFIG)
            try:
                for fact in ('Luna é um gato.','Todo gato é um mamífero.','Todos os mamíferos são animais.'):
                    brain.chat(fact)
                agent=Autonomy(brain,AutoSettings(training=True,experiments=True,growth=False,research=False))
                agent.tick();agent.tick()
                count=brain.memory.db.execute("SELECT COUNT(*) FROM lessons WHERE origin='reasoning_proof'").fetchone()[0]
                self.assertGreater(count,0)
                self.assertTrue(any(kind=='reasoning_practice' for kind,_,_ in brain.memory.recent_events(10)))
            finally:brain.close()

    def test_teaching_in_conversation_and_reasoning(self):
        with tempfile.TemporaryDirectory() as temp:
            brain=Brain(Path(temp),start_autonomy=False)
            brain.core=NeuralCore(Path(temp)/'brain-v2.pt',CONFIG)
            try:
                brain.chat('Luna é um gato.')
                brain.chat('Todo gato é um mamífero.')
                brain.chat('Todos os mamíferos são animais.')
                answer,source=brain.chat('Luna é um animal?')
                self.assertIn('Sim.',answer)
                self.assertIn('simbólico',source)
                self.assertEqual(brain.memory.reasoning_fact_count(),3)
                math,_=brain.chat('Quanto é 2 + 3 * 4?')
                self.assertIn('14',math)
                self.assertEqual(brain.core.steps,3)
            finally:brain.close()
            reopened=Brain(Path(temp),start_autonomy=False)
            try:self.assertIn('Sim.',reopened.chat('Luna é um animal?')[0])
            finally:reopened.close()

if __name__=='__main__':unittest.main()
