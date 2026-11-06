"""Isolated basic PT-PT instruction tuning experiment on Nord's Transformer.

The script starts from a copy of an existing language checkpoint, trains only
the original Nord Transformer, and never writes to the supplied checkpoint.
Held-out paraphrases are not included in the training pairs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nord_brain.core import NeuralCore
from nord_brain.candidate_evaluation import choose_candidate


TRAIN = (
    ("Ol\u00e1, como est\u00e1s?", "Ol\u00e1! Estou por aqui e pronto para conversar."),
    ("O que \u00e9 uma \u00e1rvore?", "Uma \u00e1rvore \u00e9 uma planta com tronco, ramos e folhas."),
    ("Para que serve a \u00e1gua?", "A \u00e1gua \u00e9 essencial para a vida e para as atividades do dia a dia."),
    ("O que faz o Sol?", "O Sol fornece luz e calor \u00e0 Terra."),
    ("O que precisam as plantas para crescer?", "As plantas precisam de luz, \u00e1gua, ar e nutrientes."),
    ("O que \u00e9 a amizade?", "A amizade \u00e9 uma rela\u00e7\u00e3o de afeto, confian\u00e7a e apoio."),
    ("Porque temos dia e noite?", "A Terra roda sobre si pr\u00f3pria; por isso, diferentes lugares recebem luz do Sol."),
    ("O que faz uma raiz?", "A raiz fixa a planta ao solo e ajuda a absorver \u00e1gua e nutrientes."),
    ("O que fazes quando n\u00e3o sabes?", "Digo que n\u00e3o sei e pe\u00e7o mais informa\u00e7\u00e3o antes de responder."),
    ("Podes explicar de outra forma?", "Claro. Posso usar palavras mais simples e dar um exemplo."),
    ("O que \u00e9 um livro?", "Um livro re\u00fane texto ou imagens para informar, ensinar ou contar uma hist\u00f3ria."),
    ("Como se resolve um problema?", "Definimos o problema, comparamos solu\u00e7\u00f5es e verificamos o resultado."),
)

# These prompts and target answers are reserved for evaluation only.
HOLDOUT = (
    ("Diz-me ol\u00e1 e pergunta como estou.", "Ol\u00e1! Como est\u00e1s?"),
    ("Que tipo de ser vivo \u00e9 uma \u00e1rvore?", "Uma \u00e1rvore \u00e9 uma planta."),
    ("Porque precisamos de beber \u00e1gua?", "A \u00e1gua \u00e9 essencial para a vida."),
    ("De onde vem a luz e o calor que recebemos durante o dia?", "O Sol fornece luz e calor \u00e0 Terra."),
    ("O que ajuda uma planta a alimentar-se e manter-se no ch\u00e3o?", "As ra\u00edzes absorvem \u00e1gua e nutrientes e fixam a planta ao solo."),
    ("Como respondes quando te falta informa\u00e7\u00e3o?", "Digo que n\u00e3o sei e pe\u00e7o mais informa\u00e7\u00e3o."),
    ("Explica a altern\u00e2ncia entre o dia e a noite.", "A rota\u00e7\u00e3o da Terra faz com que diferentes lugares recebam luz do Sol."),
    ("O que torna duas pessoas amigas?", "A amizade envolve afeto, confian\u00e7a e apoio."),
)


def run(checkpoint: Path, cycles: int, *, threads: int = 4,
        rate: float = 0.00002, anchor_strength: float = 0.01) -> dict:
    import torch

    torch.set_num_threads(threads)
    with tempfile.TemporaryDirectory(prefix="nord-basic-experiment-") as folder:
        trial = Path(folder) / "language-v1.pt"
        shutil.copy2(checkpoint, trial)
        core = NeuralCore(trial)
        anchor = {name: param.detach().clone()
                  for name, param in core.model.named_parameters()}
        before = core.steps
        eval_pairs = [("Tu: " + question, answer) for question, answer in HOLDOUT]
        holdout_before = core.score(eval_pairs)
        known_before = core.score([("Tu: " + question, answer)
                                   for question, answer in TRAIN])
        for _ in range(cycles):
            for question, answer in TRAIN:
                core.train("Tu: " + question, answer, epochs=1,
                           rate=rate, dialogue=True)
                if anchor_strength:
                    import torch
                    with torch.no_grad():
                        for name, param in core.model.named_parameters():
                            param.lerp_(anchor[name], anchor_strength)
        holdout_after = core.score(eval_pairs)
        known_after = core.score([("Tu: " + question, answer)
                                  for question, answer in TRAIN])
        exact = []
        for question, answer in TRAIN:
            raw = core.generate("Tu: " + question, max_bytes=115,
                                temperature=0, dialogue=True)
            exact.append({"prompt": question, "expected": answer,
                          "generated": raw, "exact": raw == answer})
        unseen = []
        for question, expected in HOLDOUT:
            candidates = [core.generate("Tu: " + question, max_bytes=115,
                                        temperature=0 if i == 0 else .25,
                                        dialogue=True)
                          for i in range(3)]
            selected = choose_candidate(candidates, question)
            unseen.append({"prompt": question, "expected": expected,
                           "candidates": candidates, "selected": selected})
        return {
            "source_checkpoint": str(checkpoint),
            "source_steps": before,
            "trial_steps": core.steps,
            "updates": core.steps - before,
            "cycles": cycles,
            "learning_rate": rate,
            "anchor_strength": anchor_strength,
            "train_loss_before": round(known_before, 4),
            "train_loss_after": round(known_after, 4),
            "heldout_loss_before": round(holdout_before, 4),
            "heldout_loss_after": round(holdout_after, 4),
            "known_exact": sum(row["exact"] for row in exact),
            "known_total": len(exact),
            "known_generations": exact,
            "unseen_accepted": sum(row["selected"] is not None for row in unseen),
            "unseen_total": len(unseen),
            "unseen_generations": unseen,
            "note": "This experiment never modifies the source checkpoint. Loss is not fluency.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cycles", type=int, default=8,
                        help="passes through the 12 training examples (default: 8)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rate", type=float, default=0.00002)
    parser.add_argument("--anchor", type=float, default=0.01,
                        help="per-update interpolation toward original weights; 0 disables")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("checkpoint not found")
    if not 1 <= args.cycles <= 100:
        parser.error("cycles must be between 1 and 100")
    if not 0 < args.rate <= 0.0003:
        parser.error("rate must be in (0, 0.0003]")
    if not 0 <= args.anchor <= 1:
        parser.error("anchor must be between 0 and 1")
    print(json.dumps(run(args.checkpoint, args.cycles, threads=args.threads,
                         rate=args.rate, anchor_strength=args.anchor),
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
