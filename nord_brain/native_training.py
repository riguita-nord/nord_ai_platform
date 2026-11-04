"""CPU training on human-approved, UTF-8 Portuguese text.

The user's own documents can serve as a source of next-byte language labels
without a pretrained external LLM. No browsing, automatic self-training or
fabricated language-quality scores. Teacher lessons are protected by rollback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path


@dataclass(frozen=True)
class CorpusResult:
    path: str
    digest: str
    bytes_used: int
    updates: int
    loss: float
    accepted: bool
    reason: str


def train_approved_file(brain, path: str | Path, *, updates: int=6) -> CorpusResult:
    if not 1 <= updates <= 32:
        raise ValueError('O treino aceita entre 1 e 32 atualizações por sessão')
    try:
        source=Path(path).expanduser().resolve(strict=True)
    except (FileNotFoundError,NotADirectoryError) as exc:
        raise ValueError('O ficheiro de treino aprovado não existe') from exc
    if not source.is_file() or source.suffix.lower() not in ('.txt','.md'):
        raise ValueError('Seleciona um ficheiro .txt ou .md com texto português aprovado')
    if source.stat().st_size > 100000:
        raise ValueError('Máximo: 100 KB de texto aprovado por sessão')
    text=source.read_text(encoding='utf-8')
    byte_count=len(text.encode('utf-8'))
    if not 180 <= byte_count <= 100000:
        raise ValueError('São necessários entre 180 bytes e 100 KB de texto')
    digest=hashlib.sha256(text.encode('utf-8')).hexdigest()

    with brain.lock:
        if brain.memory.training_paused():
            raise ValueError('A aprendizagem está pausada')
        # Older teacher notes are the most authoritative model-specific
        # examples; scoring protects them against catastrophic forgetting.
        teacher=brain.memory.db.execute(
            "SELECT question,answer FROM lessons WHERE origin='teacher' "
            "ORDER BY id DESC LIMIT 4"
        ).fetchall()
        probes=brain.memory.tests(4)
        before_teacher=brain.core.score(teacher) if teacher else None
        before_probes=brain.core.score(probes) if probes else None
        old_steps=brain.core.steps
        brain.core.save()
        try:
            loss=brain.core.train_text(text,updates=updates)
            after_teacher=brain.core.score(teacher) if teacher else None
            after_probes=brain.core.score(probes) if probes else None
            accepted=math.isfinite(loss)
            if before_teacher is not None:
                accepted=accepted and math.isfinite(after_teacher) and (
                    after_teacher <= before_teacher*1.08+0.01)
            if before_probes is not None:
                accepted=accepted and math.isfinite(after_probes) and (
                    after_probes <= before_probes*1.12+0.01)
            if accepted:
                brain.core.save()
                reason='Treino aceite; pesos e otimizador guardados'
            else:
                brain.core.load()
                reason='Regressão detetada; checkpoint anterior recuperado'
            gained=brain.core.steps-old_steps
            brain.memory.event('approved_text_training',
                               f'{source.name[:75]} sha256={digest[:16]}; '
                               f'accepted={accepted}; updates={gained}',float(gained))
            return CorpusResult(str(source),digest,byte_count,
                                gained,float(loss),accepted,reason)
        except Exception:
            brain.core.load()
            raise


def describe(result: CorpusResult) -> str:
    return (f'{result.reason}. Ficheiro: {result.path}. '
            f'Bytes aprovados: {result.bytes_used}; passos reais: {result.updates}; '
            f'perda de treino: {result.loss:.4f}. '
            'Isto mede aprendizagem de sequências de texto; não prova '
            'compreensão nem fluência em novas conversas.')
