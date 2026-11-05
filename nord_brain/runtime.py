"""The Nord Brain terminal runtime: conversational teaching and autonomous training.

Responses distinguish symbolic knowledge from unverified neural generation.
Curated official-source excerpts are tracked separately from verified facts.
"""
from __future__ import annotations
from dataclasses import asdict
import os
import re
from pathlib import Path
import threading

from .core import NeuralCore, TransformerConfig
from .storage import Memory
from .vocabulary import initialize
from .conversation import Intent, interpret, is_self_knowledge_request, self_query_kind
from .self_knowledge import report as self_knowledge_report, respond as self_knowledge_respond
from .natural_math import read_math_question, read_math_example
from .hypotheses import propose, explore_theories
from .autonomy import Autonomy
from .internet import search, OFFICIAL, public_question
from .reasoning import ReasoningEngine, parse_fact
from .world import WorldMind
from .theory_control import route_control
from .cognition import CognitiveLab
from .creative import creative_request, story
from .language_backend import LocalLanguage  # legacy, explicit opt-in only
from .native_language import NativeLanguage
from .intent_learning import NeuralIntent
from .context_intent import ContextIntent
from .social_dialogue import social_act, social_response, conversational_answer
from .story_studio import StoryStudio, BookCommand, parse_book_command
from .dialogue import DialogueMemory, followup_kind
from .story_revision import revise_story


class Brain:
    def __init__(self,data_dir: Path, *, start_autonomy: bool = True):
        self.data_dir=Path(data_dir)
        self.data_dir.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()
        self.memory=Memory(self.data_dir/'memory.sqlite3')
        self.vocabulary=initialize(self.data_dir/'vocabulary.json')
        profiles={
            'tiny':TransformerConfig(d_model=64,layers=2,heads=2,kv_heads=1,intermediate_size=192,context=128),
            'small':TransformerConfig(),
            'medium':TransformerConfig(d_model=192,layers=6,heads=6,kv_heads=3,intermediate_size=512,context=384),
        }
        profile=os.environ.get('NORD_BRAIN_PRESET','small').lower()
        if profile not in profiles:
            raise ValueError('NORD_BRAIN_PRESET deve ser tiny, small ou medium')
        self.core=NeuralCore(self.data_dir/'brain-v2.pt',config=profiles[profile])
        # Upgrade existing learned weights rather than replacing the checkpoint.
        # Keep a one-time backup for rollback; depth expansion is opt-in for tiny.
        if self.core.model.config.layers < 8:
            import shutil
            checkpoint = self.data_dir/'brain-v2.pt'
            backup = self.data_dir/'brain-v2.pre-8layers.pt'
            if checkpoint.exists() and not backup.exists():
                shutil.copy2(checkpoint, backup)
            self.core.expand_depth(8)
            self.core.save()
        # Fixed balanced context for CPU; RoPE has no learned position table.
        if self.core.model.config.context != 512:
            self.core.model.config.context = 512
            self.core.save()
        # Independent language weights: autonomous mathematics must not overwrite
        # the conversational model. Copy the existing checkpoint once for
        # migration; never delete lessons, dialogue or the original weights.
        language_checkpoint = self.data_dir/'language-v1.pt'
        if not language_checkpoint.exists():
            import shutil
            shutil.copy2(self.data_dir/'brain-v2.pt', language_checkpoint)
        self.language_core=NeuralCore(language_checkpoint, config=self.core.model.config)
        self.reasoner=ReasoningEngine(self.memory)
        # Separate CPU-trainable predictive brain for sensorimotor experiences.
        # Its weights are trained from transitions, never injected via chat context.
        self.world=WorldMind(self.data_dir/'world-model-v1.pt')
        self.cognition=CognitiveLab(self)
        # The Nord Transformer is the primary language engine. An independent
        # pretrained model is never started by default; the legacy adapter
        # requires an explicit developer opt-in.
        self.language=(LocalLanguage() if os.getenv('NORD_LANGUAGE_BACKEND','native').strip().lower()=='external'
                       else NativeLanguage(self))
        self.stories=StoryStudio(self.data_dir)
        self.dialogue=DialogueMemory(self.data_dir)
        # Neural classification of contextual revisions, separate from facts.
        # This small supervised head is part of Nord Brain (no external LLM).
        self.intent=NeuralIntent(self.data_dir)
        self.context_intent=ContextIntent(self.data_dir)
        # Every reply, including symbolic reasoning and web research, now
        # becomes context for the local language model and survives restart.
        self.language.context_provider=self.dialogue.context_messages
        self._greeting_index=-1
        self.last_question=None
        self.last_research=None
        self.awaiting_explanation=False
        self.autonomy=Autonomy(self)
        if start_autonomy:self.autonomy.start()

    def _math_reply(self, solved):
        """Neural prediction first, separate exact verification for correctness.

        Only a correct neural prediction counts as learned. The arithmetic
        verifier is an external safety fallback, never a neural achievement.
        """
        expr, proof = solved
        actual = proof.steps[-1].split(' = ')[-1]
        prompt = f'Quanto é {expr}?'
        # Use the same canonical format used for independent training/evaluation.
        predicted = self.core.generate(prompt,max_bytes=8,temperature=0).strip()
        correct = predicted == actual
        self.memory.metric('math_neural_attempt',float(correct))
        origin = ('cálculo verificável; tentativa neuronal correta' if correct else
                  'cálculo verificável; tentativa neuronal ainda incorreta')
        return proof.answer, origin

    def _checked_math(self,message: str):
        """Work examples are verified before training; questions do not get learnt as labels."""
        example=read_math_example(message)
        if example is not None:
            self.last_question=None
            self.awaiting_explanation=False
            if not example.verified:
                return (f'A conta não confirma {example.claimed}. {example.proof.answer} '
                        'Não guardei a afirmação incorreta.'), 'verificador matemático'
            question=f'Quanto é {example.expression}?'
            if not self.memory.is_holdout(question) and self.memory.exact(question)!=example.actual:
                self._learn(question,example.actual,epochs=1,origin='verified_math_example')
            # An example is the beginning of a hypothesis-search task. These
            # NEW expressions are verified and trained during autonomous ticks,
            # not inserted into the chat context as if already understood.
            # Record counterfactual reasoning first, so the evidence explaining
            # a hypothesis is retained even when generic exercises overlap.
            theories=explore_theories(example.expression,example.actual)
            queued_theories=self.memory.queue_theories(example.expression,theories)
            queued=self.memory.queue_math_hypotheses(
                example.expression, propose(example.expression, example.actual))
            self.memory.event('hypothesis_seed',
                              f'{example.expression}={example.actual}; queued={queued}; theories={queued_theories}',queued+queued_theories)
            return (f'Correto: {example.proof.answer} '
                    f'Registei o exemplo verificado. Preparei {queued+queued_theories} hipóteses, incluindo acrescentar e retirar números, para verificar e aprender '
                    'nos próximos ciclos. Ainda não considero isso compreensão matemática.'), 'cálculo verificado + laboratório de hipóteses'
        solved=read_math_question(message)
        if solved is not None:
            self.last_question=None
            self.awaiting_explanation=False
            return self._math_reply(solved)
        return None

    def absorb_official_sources(self, topic, results):
        """Archive public evidence only; never promote a webpage to teacher truth.

        Even official publications can be incomplete, outdated or unsuitable.
        Human-approved knowledge outranks external discovery.
        """
        for item in results[:3]:
            if item.get('source_tier')=='official_primary':
                self.memory.observation(topic,item.get('title',''),
                                        item.get('url',''),item.get('summary',''))
        return 0  # deliberately no automatic neural training from research

    def _research_public(self,topic: str, question: str):
        """Research is evidence discovery, not free-form neural generation.

        Returns links and snippets without asserting that one webpage is truth.
        Sources are not trained until the teacher approves a specific result.
        """
        self.awaiting_explanation=False
        self.last_question=question
        if os.getenv('NORD_INTERNET','on').lower() in ('off','0','false','no'):
            return ('A Internet está desativada (NORD_INTERNET=off). Não vou inventar uma resposta.',
                    'sem Internet')
        results=search(topic)
        self.last_research=None
        if not results:
            self.memory.unknown(topic)
            return ('Não encontrei fontes úteis sobre esse tema. A pesquisa pública pode estar indisponível '
                    'na rede, ou a fonte não apareceu nos resultados. Não vou inventar uma resposta. '
                    'Podes ensinar-me um facto ou executar /diagnostico-pesquisa para testar os fornecedores. Para maior fiabilidade, configura NORD_BRAVE_API_KEY.'), 'pesquisa sem resultados'
        # Preserve the former opt-in research pipeline for the four strictly
        # pinned official documentation pages. Broader search snippets are
        # NEVER admitted as autonomous neural labels without teacher review.
        official_training=self.absorb_official_sources(topic,results)
        for item in results:
            if item.get('source_tier')!='official_primary':
                self.memory.observation(topic,item['title'],item['url'],item['summary'])
        self.last_research={'question':question,'topic':topic,'results':results}
        lines=[f'Encontrei estas informações sobre {topic} (ainda não confirmadas):']
        if results[0].get('source_tier')=='official_primary':
            lines.append('Fonte oficial encontrada; informação provisória, ainda sem treino.')
        for index,item in enumerate(results[:3],1):
            lines.append(f'[{index}] {item["title"]}: {item["summary"][:290]}')
            lines.append(f'    Fonte: {item["url"]}')
        lines.append('Estes são excertos de pesquisa, não conhecimentos aprendidos nos pesos. '
                     'Para validares uma fonte e ensiná-la, diz «confirma fonte 1»; '
                     'para corrigir, explica-me o facto correto.')
        return '\n'.join(lines),'pesquisa pública com fontes; ainda não validada'

    def _confirm_public(self,index: int):
        if not self.last_research:
            return ('Não tenho uma pesquisa pendente para confirmar. Pesquisa primeiro o tema.',
                    'sem pesquisa pendente')
        record=self.last_research
        if not 1<=index<=len(record['results']):
            return ('Esse número de fonte não existe na última pesquisa.', 'fonte desconhecida')
        item=record['results'][index-1]
        answer=item['summary'][:280].strip()
        if len(answer)<18:
            return ('Esta fonte não contém informação suficiente para aprender.', 'fonte insuficiente')
        # The user explicitly validates the selected excerpt. The content
        # is still attributed; train a real gradient update, NOT prompt context.
        question=record['question']
        # An explicit teacher answer cannot be silently superseded by an
        # extracted snippet, even when the user approves that snippet generally.
        old=self.memory.db.execute('''SELECT answer FROM lessons WHERE lower(question)=lower(?)
                       AND origin='teacher' ORDER BY id DESC LIMIT 1''',(question,)).fetchone()
        if old and old[0].strip()!=answer:
            return ('A fonte contradiz uma resposta que me ensinaste diretamente. '
                    'Vou manter a tua resposta. Se quiseres alterá-la, corrige-a explicitamente.'), 'conflito com o professor'
        if self.memory.is_holdout(question) or self.memory.is_cognitive_probe(question):
            return ('Esta pergunta está reservada para avaliação, por isso não posso treiná-la.', 'teste protegido')
        self._learn(question,answer,epochs=1,origin='teacher_confirmed_research')
        topic=record['topic']
        if len(topic)<=60:
            self.memory.concept(topic,answer[:300],origin='teacher_confirmed_research')
        self.memory.event('research_confirmed',item['url'])
        self.last_research=None
        return (f'Confirmação registada. Atualizei os pesos com a informação que validaste sobre {topic}. '
                'O excerto continua associado à fonte '+item['url']+'. '
                'Isto é treino real, mas ainda não demonstra compreensão ou generalização.'), 'revisão do utilizador + treino neuronal'

    def _learn(self,question,answer,epochs=1,origin='teacher'):
        """Actual online gradient update followed by durable persistence."""
        question,answer=question.strip(),answer.strip()
        if not 1<=len(question)<=250 or not 1<=len(answer)<=500:
            raise ValueError('Exemplo vazio ou demasiado longo (250/500 caracteres)')
        if self.memory.is_training_blocked(question):
            raise ValueError('Treino desta pergunta bloqueado pelo utilizador')
        if self.memory.is_holdout(question) or self.memory.is_cognitive_probe(question):
            raise ValueError('Este exemplo está reservado para avaliação independente')
        self.core.save()
        try:
            loss=self.core.train(question,answer,epochs=epochs)
            self.core.save()
        except Exception:
            self.core.load()
            raise
        # Commit the example only after neural training succeeded.
        self.memory.teach(question,answer,origin=origin)
        self.memory.metric('train_loss',loss)
        return loss

    def teach(self,question,answer,epochs=2):
        with self.lock:return self._learn(question,answer,epochs=epochs)

    def answer(self,question,neural_only=False):
        with self.lock:
            question=question.strip()
            self.last_question=question
            if not neural_only:
                self_kind=self_query_kind(question)
                if self_kind is not None:
                    if self_kind == 'unknown':
                        taught=self.memory.exact(question)
                        if taught is not None:
                            return taught, 'memória ensinada sobre a própria Nord Brain'
                    self.last_question=None
                    return self_knowledge_respond(self,self_kind), 'autoconhecimento local verificado'
                solved=read_math_question(question)
                if solved is not None:
                    return self._math_reply(solved)
                if is_self_knowledge_request(question) or self.memory.resolve_intent_alias(question)=='self_knowledge':
                    return self_knowledge_report(self.memory,self.core, internet_enabled=os.getenv('NORD_INTERNET','on').lower() not in ('off','0','false','no'),
                                                  local_language_status=self.language.status()), 'relatório local de capacidades e memória'
                known=self.memory.exact(question)
                if known is not None:
                    return known,'memória ensinada (não é generalização neuronal)'
                concept=self.memory.lookup_concept(question)
                if concept:
                    return concept,'conceito na memória'
            if self.memory.lesson_count()<15 and not neural_only:
                return 'Ainda não sei responder. Podes explicar-me?','sem treino suficiente'
            return (self.core.generate(question) or 'Ainda não sei responder.'),'rede neuronal experimental, não verificada'

    def chat(self, message: str) -> tuple[str, str]:
        """Conversation entry point: resolve references before intent/teaching.

        A dialogue reply is *not* a lesson and never updates model weights.
        Keeping a short durable working context prevents "faz uma mais longa"
        from becoming a meaningless new question or a false teaching example.
        """
        with self.lock:
            previous = self.dialogue.latest()
            # A correction of the assistant is feedback, never a new lesson
            # or a request to elaborate its last (possibly broken) answer.
            from .dialogue import normalize
            normalized = normalize(message).strip()
            correction = (normalized.startswith(('ta mal', 'esta mal', 'isso esta errado',
                          'nao era isso', 'nao e isso', 'resposta errada', 'respondeste mal'))
                          or ('nao' in normalized and 'responder' in normalized
                              and 'isso' in normalized))
            if correction:
                self.awaiting_explanation = False
                self.last_question = None
                reply = ('Percebi que a minha resposta anterior não foi adequada. '
                         'Não vou repeti-la nem tratá-la como uma lição. '
                         'Podes indicar como esperavas que eu respondesse?')
                self.dialogue.add(message, reply, 'feedback de conversa (sem treino)',
                                  'dialogue', message.strip()[:240])
                return reply, 'feedback de conversa (sem treino)'
            # Native-first for ordinary conversation. Explicit commands, factual
            # requests, teaching and corrections retain their verified routes.
            # No generated response is ever fed back as a teacher label.
            if (previous and not message.lstrip().startswith('/')
                    and len(message) <= 240
                    and read_math_question(message) is None
                    and read_math_example(message) is None
                    and parse_book_command(message) is None
                    and creative_request(message) is None
                    and self_query_kind(message) is None):
                conversational_intent = interpret(message, None, False)
                if conversational_intent.kind not in (
                        'fact', 'teach', 'identity', 'correction', 'math_question',
                        'math_example', 'search', 'intent_instruction'):
                    generated = self.language.generate(message, max_tokens=180)
                    if generated:
                        self.awaiting_explanation = False
                        self.last_question = None
                        self.dialogue.add(message, generated,
                                          'geração neuronal contextual', 'dialogue',
                                          previous.get('topic') or message[:240])
                        return generated, 'geração neuronal contextual'
            # If a conversational completion fails, report model limitations
            # rather than misclassifying the utterance as a missing fact.
            # This is a diagnostic, never a fabricated conversational reply.
            if (previous and not message.lstrip().startswith('/')
                    and read_math_question(message) is None
                    and read_math_example(message) is None
                    and parse_book_command(message) is None
                    and creative_request(message) is None):
                parsed = interpret(message, None, False)
                if parsed.kind not in (
                        'fact', 'teach', 'identity', 'correction', 'math_question',
                        'math_example', 'search', 'intent_instruction',
                        'definition_question', 'explanation'):
                    reply = ('A minha geração neuronal ainda não conseguiu produzir '
                             'uma resposta conversacional fiável. Preciso de mais '
                             'treino linguístico, não de uma lição factual.')
                    self.awaiting_explanation = False
                    self.last_question = None
                    self.dialogue.add(message, reply, 'falha de geração neuronal',
                                      'dialogue', message[:240])
                    return reply, 'falha de geração neuronal'
            # Do not force arbitrary short messages into a scripted fallback.
            # Let contextual intent and dialogue generation evaluate the turn.
            kind = followup_kind(message)
            # Neural head interprets unexpected phrasing against the active
            # subject; it can override a generic heuristic, but never an
            # explicit new task, a verified equation or a teacher lesson.
            if previous and kind in (None,'discuss','refine') and not (
                    creative_request(message) is not None
                    or parse_book_command(message) is not None
                    or read_math_question(message) is not None
                    or read_math_example(message) is not None
                    or is_self_knowledge_request(message)):
                intent_guard=interpret(message,None,False)
                if intent_guard.kind not in (
                        'fact','teach','identity','correction','math_question',
                        'math_example','search','intent_instruction') and (
                        self_query_kind(message) in (None,'unknown')):
                    suggested,confidence=self.intent.predict(
                        message,previous['focus'])
                    if suggested is not None and (kind is None or
                            suggested not in ('discuss','explain')):
                        kind=suggested
            result = None
            if previous and kind is not None:
                result = self._contextual_reply(message, kind, previous)
                if result is not None:
                    # A continuation or discussion request is never a lesson
                    # answering an earlier unresolved question.
                    self.awaiting_explanation=False
                    self.last_question=None
            if result is None:
                result = self._chat_impl(message)
            reply, origin = result
            if origin.startswith(('livro persistente', 'estúdio de histórias',
                                  'plano narrativo', 'biblioteca local',
                                  'manuscrito guardado')):
                focus = 'book'
            elif origin.startswith(('composição narrativa', 'história contextual',
                                    'ficção contextual')) or (creative_request(message) is not None
                    and not creative_request(message).startswith('IDEIAS:')):
                focus = 'story'
            elif kind is not None and previous is not None and result is not None and (
                    origin.startswith('continuidade contextual')
                    or origin.startswith('ficção contextual')):
                focus = previous['focus']
            elif 'cálculo' in origin or 'matemátic' in origin:
                focus = 'math'
            elif 'pesquisa' in origin or 'fonte' in origin:
                focus = 'research'
            else:
                focus = 'dialogue'
            # Short follow-ups inherit the subject; a new independent request
            # begins a new topic. This topic is a conversation bookmark only,
            # not a verified fact or learned neural weight.
            if kind is not None and previous is not None:
                topic=previous.get('topic') or previous['question']
            else:
                topic=message.strip()[:240]
            self.dialogue.add(message, reply, origin, focus, topic)
            return reply, origin

    def _contextual_reply(self, message: str, kind: str,
                          previous: dict[str,str]) -> tuple[str,str] | None:
        """Respond about the previous answer rather than memorising the prompt."""
        focus = previous['focus']
        if kind == 'chapters':
            if focus == 'book':
                book=self.stories.active()
                if book:
                    return (f'A história «{book["title"]}» já está organizada em '
                            f'capítulos ({book["written"]}/{book["planned"]}). '
                            'Diz «continua a história» para escrever o seguinte, '
                            'ou «plano do livro» para consultar a estrutura.'), \
                           'livro persistente; contexto reconhecido'
            if focus == 'story':
                story_turn=self.dialogue.latest_story()
                original=story_turn['reply'] if story_turn else previous['reply']
                try:
                    adopted=self.stories.adopt_story(original,chapters=10)
                except ValueError:
                    adopted=None
                if adopted is not None:
                    parts='\n\n'.join(
                        f'Capítulo {ch["number"]} — {ch["title"]}\n\n{ch["body"]}'
                        for ch in adopted['chapters'])
                    return (f'{adopted["book"]["title"]}\n\n{parts}\n\n'
                            f'Guardei este enredo como um livro com '
                            f'{adopted["book"]["planned"]} capítulos planeados. '
                            'As primeiras partes preservam a história que já criámos. '
                            'Diz «continua a história» para desenvolver o capítulo '
                            'seguinte, ou «plano do livro» para ver a estrutura.'), \
                           'livro persistente; história da conversa organizada em capítulos'
        if kind == 'refine' and focus == 'story':
            story_turn=self.dialogue.latest_story()
            original=story_turn['reply'] if story_turn else previous['reply']
            prompt=('Reformula a MESMA história em português europeu, respeitando '
                    'rigorosamente esta alteração pedida pelo utilizador: '
                    +message[:200]+'. Preserva os nomes e os acontecimentos que '
                    'já existem. Responde apenas com a prosa revista.\n'
                    'HISTÓRIA ANTERIOR:\n'+original[:5200])
            output=self.language.generate(prompt,creative=True,max_tokens=1200,
                                          remember=False)
            if output and len(output.strip())>=len(original)*0.65:
                return output,'ficção contextual revista pelo Transformer Nord (experimental)'
            # The offline generator can add development/dialogue, but cannot
            # promise arbitrary transformations of tone, register or style.
            from .dialogue import normalize
            instruction=normalize(message)
            if any(word in instruction for word in
                   ('detalh', 'dialog', 'desenvolv', 'cenas', 'mais texto')):
                return revise_story(original,'longer'), \
                       'ficção contextual procedural (expansão limitada)'
            return ('Continuamos na mesma história. Percebi a alteração: «'
                    +message[:180]+'». Não consigo reescrever livremente o estilo '
                    'quando a geração neuronal ainda não estiver fluente. Podes pedir uma versão mais '
                    'longa, capítulos ou continuação, que consigo fazer offline.'), \
                   'continuidade contextual com recursos neuronais ainda insuficientes'
        if kind == 'reference' and focus in ('story','book'):
            original = previous['reply']
            if focus == 'book':
                book = self.stories.active()
                row = self.stories.chapter(book['written']) if book and book['written'] else None
                original = row[1] if row else original
            prompt = ('Responde à pergunta sobre a história anterior, em português '
                      'e sem inventar acontecimentos.\nPergunta: '+message[:220]
                      +'\nHistória:\n'+original[:5500])
            result = self.language.generate(prompt, creative=False, max_tokens=240,
                                            remember=False)
            if result:
                return result,'continuidade contextual com Transformer Nord (experimental)'
            from .story_revision import _elements
            name, where, discovery = _elements(original)
            from .dialogue import normalize
            form = normalize(message)
            if 'nome' in form or 'quem' in form or 'personagem' in form:
                return f'A personagem principal da história é {name}.', 'continuidade contextual da história'
            if 'onde' in form:
                return f'Na história, {name} vivia {where}.', 'continuidade contextual da história'
            if 'encontr' in form:
                return f'Na história, {name} encontrou {discovery}.', 'continuidade contextual da história'
        if focus == 'book':
            if kind == 'continue':
                return self.book_operation(BookCommand('continue'))
            if kind in ('longer','rewrite','ending'):
                output = self.stories.revise_last(self.language, kind=kind)
                if output is not None:
                    return (f'Capítulo {output["number"]} — {output["title"]} '
                            f'(versão revista)\n\n{output["body"]}\n\n'
                            'A alteração ficou guardada no livro.'), 'livro persistente; capítulo revisto'
        if focus == 'story' and kind in ('longer','rewrite','ending','continue'):
            # A follow-up about a character may have been the last reply; find
            # the most recent full narrative instead of expanding that answer.
            story_turn=self.dialogue.latest_story()
            source_story=story_turn['reply'] if story_turn is not None else previous['reply']
            instructions = {
                'longer': 'Reescreve a MESMA história numa versão significativamente mais longa, '
                          'com novas cenas, diálogos, desenvolvimento e detalhes. Preserva '
                          'personagens, objetos, conflitos e acontecimentos anteriores.',
                'rewrite': 'Reescreve esta história de outra forma, mas mantém os nomes, '
                           'o tema, os acontecimentos essenciais e a continuidade.',
                'ending': 'Mantém a história até ao momento da resolução e escreve um '
                          'final alternativo coerente com os mesmos personagens.',
                'continue': 'Escreve a continuação desta MESMA história. Não recontes o '
                            'início, avança os acontecimentos e preserva a continuidade.'
            }
            prompt = (instructions[kind]+' Responde em português europeu, em prosa. '
                      'Este texto é ficção, não informação factual.\n'
                      'HISTÓRIA ANTERIOR:\n'+source_story[:5600])
            output = self.language.generate(prompt, creative=True, max_tokens=1200,
                                            remember=False)
            if (not output or (kind == 'longer'
                               and len(output.strip()) < len(source_story)*1.25)):
                output = revise_story(source_story,kind)
                return output, 'ficção contextual procedural (sem modelo pré-treinado)'
            return output, 'ficção contextual gerada pelo Transformer Nord (experimental)'
        if kind in ('explain','discuss','reference','shorter','longer','rewrite','continue','refine','chapters'):
            prompt = (
                'Continua a conversa em português europeu. Usa o histórico recente '
                'e mantém o assunto. Interpreta «ele», «isso» ou pedidos de revisão '
                'com base nas mensagens anteriores. Não memorizes a pergunta como '
                'resposta, não inventes factos nem fontes, e respeita o ensino do '
                'utilizador. Se faltar informação, pede uma clarificação curta. '
                'Pedido atual: ' + message[:300])
            output = self.language.generate(prompt,creative=False,max_tokens=430)
            if output:
                return output,'continuidade contextual com Transformer Nord (experimental)'
            # Avoid claiming the scratch-trained Transformer understands a
            # reference to material it has not learned to paraphrase.
            # Do not echo the previous answer into the next answer: it
            # causes exponential repetition on "faz isso" / "explica melhor".
            return ('Percebi que queres continuar ou reformular a resposta. '
                    'Ainda não consigo gerar essa explicação com fiabilidade. '
                    'Podes indicar o ponto específico que queres desenvolver?'), \
                   'continuidade contextual com recursos neuronais ainda insuficientes'
        return None

    def _chat_impl(self,message: str) -> tuple[str,str]:
        """Teacher-facing conversation. Intents are explicit, limited heuristics."""
        with self.lock:
            if message.strip().startswith('/importar-corpus '):
                from .corpus_import import import_text
                url=message.strip().split(None,1)[1]
                try:
                    path,metadata=import_text(url,self.data_dir/'corpus_importado')
                except (ValueError,OSError) as exc:
                    return f'Importação recusada: {exc}', 'importação de corpus'
                return (f'Fonte descarregada para {path}. Ainda não foi treinada. '
                        f'Revê o ficheiro e usa /treinar-linguagem 48 {path} '
                        f'para aprovar o treino.'), 'importação de corpus'
            control=route_control(self,message)
            if control is not None:
                self.awaiting_explanation=False
                self.last_question=None
                return control
            # Long fiction is persisted chapter by chapter and is never
            # confused with teacher knowledge or neural training examples.
            book_command=parse_book_command(message)
            if book_command is not None:
                self.awaiting_explanation=False
                self.last_question=None
                return self.book_operation(book_command)
            # Creative requests are not questions awaiting teacher facts.
            # Prefer pretrained local CPU inference; retain a clearly labelled
            # procedural fallback if no local language model is installed.
            creative_topic=creative_request(message)
            if creative_topic is not None:
                self.awaiting_explanation=False
                self.last_question=None
                response=self.language.generate(message,creative=True,max_tokens=650)
                if response:
                    return response,'geração neuronal local (Transformer Nord, experimental)'
                return story(creative_topic),'composição narrativa procedural (sem modelo pré-treinado)'
            # First-person questions concern the actual running agent, not
            # numbers, web research, or another teacher answer.
            self_kind=self_query_kind(message)
            if self_kind is not None:
                if self_kind == 'unknown':
                    taught=self.memory.exact(message)
                    if taught is not None:
                        self.awaiting_explanation=False
                        self.last_question=None
                        return taught, 'memória ensinada sobre a própria Nord Brain'
                    self.last_question=message.strip()
                    self.awaiting_explanation=True
                else:
                    self.awaiting_explanation=False
                    self.last_question=None
                return self_knowledge_respond(self,self_kind), 'autoconhecimento local verificado'
            # Unified hypothesis lab works across independently verifiable
            # arithmetic, patterns, logic, teacher language and virtual actions.
            lab = re.fullmatch(
                r'(?:experimenta|experimentar|simula|simular|testa|treina)\s+'
                r'(tudo|matem[aá]tica|padr[oõ]es|l[oó]gica|linguagem|mundo)'
                r'(?:\s+(\d{1,2}))?[.?!]?',message.strip(),re.I)
            if lab:
                aliases={'matemática':'matematica','padrões':'padroes','lógica':'logica'}
                domain=aliases.get(lab.group(1).casefold(),lab.group(1).casefold())
                return self.simulate_cognition(domain,int(lab.group(2) or '1'))
            # A virtual-world experiment is not a scripted answer: it actually
            # imagines four actions, executes one, and trains the world weights.
            sim = re.fullmatch(
                r'(?:simula|simular|faz uma simulação|testa hipóteses|testa hipoteses|experimenta cenários)'
                r'(?:\s+(\d{1,2})(?:\s+(?:cenários|cenarios|simulações|simulacoes|mundos))?)?[.?!]?',
                message.strip(), re.I)
            if sim:
                scenarios = min(10, max(1, int(sim.group(1) or '3')))
                return self.simulate_world(scenarios)
            math=self._checked_math(message)
            if math is not None:
                return math
            confirmation=re.fullmatch(r'(?:confirma|aprova|guarda)\s+(?:a\s+)?fonte\s+(\d{1,2})[.!]?',message.strip(),re.I)
            if confirmation:
                return self._confirm_public(int(confirmation[1]))
            topic=public_question(message)
            if topic and not is_self_knowledge_request(message):
                concept=self.memory.lookup_concept(topic)
                if concept:
                    return concept,'conceito ensinado (memória)'
                stored=self.memory.exact(message)
                if stored is not None:
                    return stored,'conhecimento validado pelo utilizador e guardado'
                return self._research_public(topic,message.strip())
            intent=interpret(message,self.last_question,self.awaiting_explanation)
            if (intent.kind in ('question','definition_question','explanation')
                    and self.memory.resolve_intent_alias(message)=='self_knowledge'):
                intent=Intent('self_knowledge')
            if intent.kind=='intent_instruction':
                self.awaiting_explanation=False
                self.last_question=None
                if intent.topic!='self_knowledge':
                    return ('Percebi que queres definir como devo interpretar uma frase, '
                            'mas ainda só consigo guardar regras sobre as minhas capacidades '
                            'e o que aprendi.'), 'instrução não suportada'
                self.memory.set_intent_alias(intent.question,'self_knowledge')
                return (f'Percebi. Quando disseres «{intent.question}», vou mostrar as '
                        'minhas capacidades e aprendizagens reais.'), 'interpretação guardada'
            if intent.kind=='self_knowledge':
                self.awaiting_explanation=False
                self.last_question=None
                return self_knowledge_respond(self,intent.topic or 'summary'), 'autoconhecimento local verificado'
            if intent.kind=='uncertain':
                return ('Tudo bem. Não vou memorizar uma dúvida como se fosse uma resposta certa. '
                        'Podes ensinar-me quando souberes?'), 'sem treino: incerteza'
            if intent.kind=='clarify_teaching':
                return ('Quero aprender corretamente, mas não consegui identificar a resposta. '
                        'Podes dar um exemplo direto, como «2+2 é 4»?'), 'a aguardar explicação clara'
            if intent.kind=='empty':return 'Diz-me alguma coisa.','terminal'
            if intent.kind=='greeting':
                self.awaiting_explanation=False
                self.last_question=None
                response=self.language.generate(message,creative=True,max_tokens=80)
                if response:return response,'geração neuronal local (Transformer Nord, experimental)'
                # Offline variation is interface behaviour, NOT evidence of
                # fluency or spontaneous neural comprehension.
                greetings=('Olá! Como estás hoje?',
                           'Olá de novo! Tens alguma ideia que queiras explorar?',
                           'Boas! O que te apetece descobrir?',
                           'Ora viva! Queres conversar ou experimentar alguma coisa?',
                           'Olá! Estou por aqui. Sobre o que queres falar?')
                self._greeting_index=(self._greeting_index+1)%len(greetings)
                return greetings[self._greeting_index],'saudação local (não neuronal)' 
            if intent.kind=='farewell':return 'Até à próxima! Vou guardar aquilo que aprendi.','interface inicial'
            if intent.kind=='progress':
                s=self.status()
                return (f"Já fiz {s['passos_de_treino']} passos de treino, tenho {s['exemplos_ensinados']} exemplos e "
                        f"{s['parametros']} parâmetros. As respostas da memória não provam raciocínio neuronal."),'métricas reais'
            if intent.kind in ('teach','identity','fact','correction','explanation'):
                self.awaiting_explanation=False
                if intent.kind=='identity':
                    self.memory.concept('nome do professor',intent.answer)
                if intent.kind=='fact':
                    self.memory.concept(intent.topic,intent.answer,origin='teacher')
                self._learn(intent.question,intent.answer,epochs=1,origin='teacher')
                if intent.kind=='fact':
                    self.reasoner.learn(message)
                if intent.kind=='correction':
                    return 'Obrigado pela correção. Atualizei os pesos e vou rever este exemplo.','aprendizagem neuronal + memória'
                if intent.kind=='identity':
                    return f'Prazer! Aprendi que te chamas {intent.answer}.','aprendizagem neuronal + memória'
                if intent.kind=='fact':
                    return f'Entendido. Aprendi a descrição de {intent.topic}.','aprendizagem neuronal + memória'
                return 'Entendido. Guardei o exemplo e atualizei os pesos neuronais.','aprendizagem neuronal + memória'
            # Unambiguous declarative teacher facts can teach relations as well.
            # The symbolic graph is distinct from the Transformer weights.
            if intent.kind=='question' and not message.strip().endswith('?'):
                fact=parse_fact(message)
                if fact:
                    self._learn(f'O que sabes sobre {fact.subject}?',fact.statement,epochs=1,origin='teacher')
                    self.reasoner.learn(message)
                    self.awaiting_explanation=False
                    return 'Aprendi essa relação. Vou tentar aplicá-la a outras perguntas.', 'premissa ensinada + treino neuronal'
            # Verified composition before a neural guess: no pretending that
            # deterministic math or graph traversal is neural generalisation.
            if intent.kind in ('question','definition_question'):
                # Existing explicitly taught answers retain priority over inference.
                direct=self.memory.exact(intent.question or message) if intent.kind=='question' else None
                if direct is not None:
                    self.last_question=message.strip()
                    self.awaiting_explanation=False
                    return direct,'memória ensinada (não é generalização neuronal)'
                # Keep the teacher's direct definition preferred over a graph listing.
                proof=(None if intent.kind=='definition_question' and self.memory.lookup_concept(intent.topic)
                       else self.reasoner.solve(message))
                if proof:
                    self.last_question=message.strip()
                    self.awaiting_explanation=proof.method=='incomplete'
                    if proof.method=='incomplete':self.memory.unknown(message[:90])
                    else:self.memory.event('reasoning',proof.method,float(len(proof.steps)))
                    return proof.answer, 'raciocínio verificável (simbólico, não neuronal)' if proof.method!='incomplete' else 'informação insuficiente'
            if intent.kind=='search':
                return self._research_public(intent.topic,message.strip())
            if intent.kind=='definition_question':
                self.last_question=f'O que é {intent.topic}?'
                known=self.memory.lookup_concept(intent.topic)
                if known:
                    self.awaiting_explanation=False
                    return known,'conceito ensinado'
                known=self.memory.exact(self.last_question)
                if known:
                    self.awaiting_explanation=False
                    return known,'memória ensinada'
                ref=self.memory.observation_for_topic(intent.topic)
                if ref:
                    return f'Encontrei uma fonte, ainda não confirmada: {ref[2][:250]}\nFonte: {ref[1]}','pesquisa não verificada'
                self.memory.unknown(intent.topic)
                self.awaiting_explanation=True
                return f'Ainda não sei o que é {intent.topic}. Podes explicar-me?', 'pergunta de aprendizagem'
            # Normal question, no teacher answer assumed. A local pretrained
            # model can converse, but never silently replace confirmed teacher
            # facts or reliable symbolic reasoning.
            self.awaiting_explanation=False
            llm_response=self.language.generate(message,max_tokens=320)
            if llm_response:
                self.last_question=None
                return llm_response,'geração neuronal local (Transformer Nord, experimental)'
            reply,source=self.answer(intent.question)
            if source=='sem treino suficiente':
                self.awaiting_explanation=True
                self.memory.unknown(intent.question[:90])
            elif source=='rede neuronal experimental, não verificada':
                # This tiny, non-pretrained LM cannot be trusted for arbitrary
                # factual text. Its raw output remains accessible in diagnostics.
                self.awaiting_explanation=True
                self.memory.unknown(intent.question[:90])
                return ('Ainda não sei responder com confiança. Podes ensinar-me, '
                        'ou pedir que pesquise fontes públicas?'), 'sem resposta factual verificada'
            return reply,source

    def book_operation(self,command: BookCommand):
        """Manage persistent manuscripts, with one bounded generation per chapter."""
        with self.lock:
            if command.action=='start':
                book=self.stories.create(command.topic,chapters=command.amount)
                output=self.stories.continue_book(self.language,1)
                chapter=output['chapters'][0]
                return (f'Comecei o livro «{book["title"]}» ({book["planned"]} capítulos).\n'
                        f'Capítulo {chapter["number"]} — {chapter["title"]}\n\n'
                        f'{chapter["body"]}\n\n'
                        'Para escrever o seguinte, diz «continua a história» ou usa /livro continuar. '
                        'O manuscrito é guardado entre sessões.'), 'livro persistente; '+chapter['source']
            if command.action=='continue':
                output=self.stories.continue_book(self.language,command.amount)
                if 'error' in output:return output['error'],'estúdio de histórias'
                if not output['chapters']:
                    return ('O livro já tem todos os capítulos planeados. '
                            'Usa /livro exportar para guardar o manuscrito.'),'estúdio de histórias'
                chapter_text='\n\n'.join(
                    f'Capítulo {c["number"]} — {c["title"]}\n\n{c["body"]}' for c in output['chapters'])
                book=output['book']
                return (chapter_text+f'\n\nLivro: {book["written"]}/{book["planned"]} capítulos escritos. '
                        'Podes continuar, reler capítulos ou exportar o manuscrito.'), 'livro persistente; '+', '.join(sorted({c['source'] for c in output['chapters']}))
            if command.action=='status':
                b=self.stories.active()
                if not b:return 'Ainda não existe um livro. Diz «cria um livro sobre ...».','estúdio de histórias'
                from .story_studio import _chapter_outline
                outline='\n'.join(f'  {n}. {_chapter_outline(b["planned"],n)[0]}'
                                  for n in range(1,b['planned']+1))
                return (f'{b["title"]} — {b["written"]}/{b["planned"]} capítulos escritos.\n'
                        f'Tema: {b["theme"]}\nPlano:\n{outline}'), 'plano narrativo persistente'
            if command.action=='list':
                books=self.stories.list_books()
                if not books:return 'Ainda não tens livros guardados.','estúdio de histórias'
                return ('Livros guardados:\n'+'\n'.join(
                    f'  #{ident}: {title} ({written}/{planned} capítulos)'
                    for ident,title,planned,written in books)), 'biblioteca local de histórias'
            if command.action=='select':
                b=self.stories.select(command.amount)
                return (f'Retomei «{b["title"]}» ({b["written"]}/{b["planned"]}).' if b
                        else 'Não encontrei esse livro.'), 'biblioteca local de histórias'
            if command.action=='read':
                row=self.stories.chapter(command.amount)
                return (f'Capítulo {command.amount} — {row[0]}\n\n{row[1]}' if row
                        else 'Esse capítulo ainda não existe no livro ativo.'), 'manuscrito guardado'
            if command.action=='export':
                output=self.stories.export()
                return (f'Manuscrito exportado em Markdown: {output}' if output else
                        'Ainda não existe um livro para exportar.'), 'ficheiro Markdown local'
            raise ValueError('Comando de livro desconhecido')

    def simulate_cognition(self, domain: str = 'tudo', examples: int = 1):
        """Train and evaluate different hypotheses with one shared protocol."""
        with self.lock:
            result=self.cognition.run(domain,examples=examples)
            return self.cognition.describe(result),'laboratório multiarea: previsões neuronais e verificadores distintos'

    def simulate_world(self, scenarios: int = 3):
        """Compare imagined actions, execute in a simulator, update real weights."""
        with self.lock:
            result=self.world.experience_cycle(scenarios=scenarios,updates=24)
            self.memory.event('world_experiment','user-directed virtual scenarios',
                              result['after']['joint_loss'])
            return self.world.describe_cycle(result),'experiência neuronal em mundo virtual (CPU)'

    def practice(self,limit=20):
        with self.lock:
            examples=self.memory.lessons(limit)
            if not examples:return 0,None
            losses=[self.core.train(q,a,epochs=1) for q,a in reversed(examples)]
            self.core.save()
            value=sum(losses)/len(losses)
            self.memory.metric('practice_loss',value)
            return len(losses),value

    def evaluate(self):
        with self.lock:
            tests=self.memory.tests(10)
            if not tests:return None
            correct=0
            for question,answer in tests:
                prediction=self.core.generate(question,temperature=0).strip().casefold()
                correct+=prediction==answer.strip().casefold()
            score=correct/len(tests)
            self.memory.metric('heldout_accuracy',score)
            return correct,len(tests),score

    def grow(self,extra=16):
        with self.lock:
            self.core.grow(extra)
            self.core.save()
            return self.core.hidden_size

    def status(self):
        with self.lock:
            return {'largura_mlp':self.core.hidden_size,
                    'arquitetura':'decoder-only Transformer (RoPE, GQA, RMSNorm, SwiGLU)',
                    'config':asdict(self.core.model.config),
                    'parametros':self.core.parameter_count,
                    'passos_de_treino':self.core.steps,
                    'exemplos_ensinados':self.memory.lesson_count(),
                    'premissas_de_raciocinio':self.memory.reasoning_fact_count(),
                    'laboratorio_de_hipoteses':self.memory.cognitive_stats(),
                    'modelo_de_mundo':{'parametros':self.world.parameter_count,
                                       'experiencias':self.world.observations,
                                       'passos_de_aprendizagem':self.world.train_steps,
                                       'cenarios':self.world.episodes},
                    'palavras_iniciais':len(self.vocabulary),
                    'testes_independentes':len(self.memory.tests(10000)),
                    'autonomia':asdict(self.autonomy.settings),
                    'eventos_autonomos':self.memory.recent_events(5),
                    'hipoteses_matematicas':self.memory.hypothesis_stats(),
                    'aprendizagem_autonoma_pausada':self.memory.training_paused(),
                    'aprovacao_de_hipoteses':self.memory.review_required(),
                    'fontes_oficiais_treinadas':self.memory.source_count(),
                    'linguagem_local':self.language.status(),
                    'intencoes_neuronais':self.intent.report(),
                    'livro_ativo':self.stories.active(),
                    'dados':str(self.data_dir)}

    def close(self):
        self.autonomy.stop()
        with self.lock:
            self.core.save()
            self.world.save()
            self.stories.close()
            self.dialogue.close()
            self.memory.close()
