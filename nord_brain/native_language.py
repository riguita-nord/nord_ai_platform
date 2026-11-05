"""Native Nord Transformer text generation, entirely offline and CPU-only.

This component uses ONLY the learnable weights in Brain.core. The dialogue
database supplies limited relevant context; it is not a second language model.
Poorly formed neural samples are rejected rather than presented as fluent text.
"""
from __future__ import annotations

import os
import re
import unicodedata


def _clean(text: str) -> str:
    return ' '.join(str(text).split())


def _compact_prompt(prompt: str, history: list[dict[str, str]]) -> str:
    """Bounded task/subject for a byte-level Transformer with a tiny window.

    Prompts involving full chapter text can exceed 5 KB. Preserve the user's
    instruction at the beginning, plus recent subjects, before the core itself
    clips old bytes to fit its checkpoint context.
    """
    stripped=_clean(prompt)
    prefix=stripped[:175]
    if len(stripped)>350:
        # The beginning has the actual command, while the tail often contains
        # the latest events of the story being continued.
        prefix=prefix[:110]+' [...] '+stripped[-95:]
    last_user=next((row['content'] for row in reversed(history)
                    if row.get('role')=='user'), '')
    last_reply=next((row['content'] for row in reversed(history)
                     if row.get('role')=='assistant'), '')
    recent=(_clean(last_user)[:44]+' '+_clean(last_reply)[:40]).strip()
    if not recent:
        return prefix
    # Limit the entire request to 170 UTF-8 bytes; training checkpoints may
    # have context=128. Never use a huge rendered history as direct token input.
    prompt=f'Assunto: {recent}. Pedido: {prefix}'
    encoded=prompt.encode('utf-8')[:172]
    return encoded.decode('utf-8','ignore')


def coherent_sample(text: str, *, creative: bool=False) -> bool:
    """Conservative presentation gate; NOT a measure of real understanding."""
    value=_clean(text)
    if len(value)<34 or len(value)>440 or len(value.split())<6:
        return False
    if '\ufffd' in value or re.search(r'[\x00-\x08\x0b-\x1f]',value):
        return False
    letters=sum(ch.isalpha() for ch in value)
    if letters/max(1,len(value)) < .66:
        return False
    words=re.findall(r'\w+',value.casefold(),re.UNICODE)
    if len(set(words))/max(1,len(words)) < .52:
        return False
    ngrams=[' '.join(words[i:i+3]) for i in range(max(0,len(words)-2))]
    if len(ngrams)>3 and len(set(ngrams))/len(ngrams)<.72:
        return False
    # Empty structured completions, control sequences and common corruption
    # in models trained only on tiny amounts of text are never shown as prose.
    if re.search(r'(?:pergunta:|resposta:|<\|end\|>|<unk>|###)',value,re.I):
        return False
    if not any(ch in value for ch in 'aeiouáàâãéêíóôõú'):
        return False
    # A byte-level model can emit plausible-looking but meaningless syllables.
    # This is only a conservative display filter, not a fluency certificate.
    common = {'a','o','os','as','um','uma','de','do','da','dos','das','e',
              'em','no','na','nos','nas','que','se','com','para','por',
              'eu','tu','ele','ela','nós','vocês','não','sim','é','são',
              'estou','está','tenho','tem','mais','mas','como','quando',
              'onde','porque','pode','podemos','quero','queres','bem',
              'olá','hoje','sobre','isso','este','esta','foi','ser'}
    recognizable = sum(word in common for word in words)
    if recognizable < 2 or recognizable / len(words) < .18:
        return False
    # Reject byte-level pseudo-Portuguese with isolated letters and
    # repeated non-lexical syllables. This is a display safeguard only.
    if sum(len(word) <= 2 for word in words) / len(words) > .32:
        return False
    if len(words) >= 8 and recognizable / len(words) < .32:
        return False
    if sum(len(word) <= 2 for word in words) / len(words) > .55:
        return False
    return True


class NativeLanguage:
    """Same small interface as the legacy backend, but NO HTTP, API or Qwen."""
    def __init__(self, brain):
        self.brain=brain
        self.context_provider=None
        self.history=[]

    def generate(self,prompt: str,*,creative: bool=False,
                 max_tokens: int=320,remember: bool=True) -> str | None:
        if os.getenv('NORD_NATIVE_GENERATION','on').strip().lower() in (
                '0','false','off','no'):
            return None
        # NORD_LLM=off disables ONLY the legacy external backend, not
        # the brain's own Transformer. To disable native generation, use
        # NORD_NATIVE_GENERATION=off explicitly.
        # A new untrained checkpoint is expected to produce noise. Avoid a
        # slow autoregressive attempt at every question until sufficient
        # actual weight updates exist. This is a budget gate, not a fluency test.
        minimum=max(0,int(os.getenv('NORD_NATIVE_MIN_STEPS','80')))
        core=self.brain.core
        if core.steps<minimum:
            return None
        history=(self.context_provider(prompt) if remember and
                 self.context_provider else [])
        from .dialogue_format import prompt_from_history
        question=prompt_from_history(prompt,history,core.model.config.context) if history else prompt
        budget=min(125,max(42,int(max_tokens)),max(42,core.model.config.context//2))
        temperature=.82 if creative else .61
        candidate=core.generate(question,max_bytes=budget,temperature=temperature)
        return candidate if coherent_sample(candidate,creative=creative) else None

    def status(self) -> str:
        return ('Nord Transformer próprio (CPU, pesos treináveis); '
                'geração experimental sem Qwen nem serviço externo')
