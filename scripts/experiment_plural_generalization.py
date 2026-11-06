"""Reproducible CPU probe: can byte-level Nord transfer regular plurals?

The held-out nouns and their forms do not occur in the training examples.
This experiment measures next-byte loss and greedy raw generation separately.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nord_brain.core import NeuralCore, TransformerConfig


def main() -> None:
    torch.set_num_threads(1)
    config = TransformerConfig(
        d_model=32, layers=2, heads=4, kv_heads=2,
        intermediate_size=64, context=128,
    )
    training = [
        ("Plural de gato?", "gatos"),
        ("Plural de livro?", "livros"),
        ("Plural de mesa?", "mesas"),
        ("Plural de casa?", "casas"),
    ]
    held_out = [
        ("Qual é o plural de bola?", "bolas"),
        ("Qual é o plural de janela?", "janelas"),
    ]
    train_nouns = {question.split("de ", 1)[1].rstrip("?") for question, _ in training}
    test_nouns = {question.rsplit("de ", 1)[1].rstrip("?") for question, _ in held_out}
    if train_nouns & test_nouns:
        raise AssertionError("Leakage between train and held-out nouns")

    with tempfile.TemporaryDirectory(prefix="nord-plural-") as directory:
        checkpoint = Path(directory) / "probe.pt"
        core = NeuralCore(checkpoint, config, seed=31)
        initial_loss = core.score(held_out)
        initial_weights = {name: parameter.detach().clone()
                           for name, parameter in core.model.named_parameters()}
        for question, answer in training:
            core.train(question, answer, epochs=12, rate=0.001)
        final_loss = core.score(held_out)
        generations = [core.generate(question, max_bytes=20, temperature=0)
                       for question, _ in held_out]
        exact = sum(got == answer for got, (_, answer) in zip(generations, held_out))
        changed_tensors = sum(
            not torch.equal(initial_weights[name], parameter)
            for name, parameter in core.model.named_parameters()
        )
        core.save()
        restored = NeuralCore(checkpoint)
        print(json.dumps({
            "seed": 31,
            "train_nouns": sorted(train_nouns),
            "held_out_nouns": sorted(test_nouns),
            "updates": core.steps,
            "parameter_tensors_changed": changed_tensors,
            "held_out_byte_loss_before": round(initial_loss, 4),
            "held_out_byte_loss_after": round(final_loss, 4),
            "raw_greedy_generations": generations,
            "held_out_exact_match": f"{exact}/{len(held_out)}",
            "persisted_steps": restored.steps,
            "checkpoint_temporary": True,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
