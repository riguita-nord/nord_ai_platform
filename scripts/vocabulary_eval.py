"""Auditable PT-PT vocabulary probes.

Byte-only scoring removes whole-word-token advantage from evaluation.
Semantic associations use explicitly held-out pairs; scores measure
generalisation and may be low. No fabricated 'known words' counts.
"""
from __future__ import annotations
from collections import Counter
import json
import re
import torch
from torch.nn import functional as F

# Manually curated PT-PT concepts; no automatic language detection claim.
TRAIN_PAIRS = [
    ("água", "líquido"), ("cão", "animal"), ("gato", "animal"),
    ("maçã", "fruta"), ("pera", "fruta"), ("carro", "veículo"),
    ("autocarro", "transporte"), ("cadeira", "móvel"),
    ("mesa", "móvel"), ("sol", "estrela"), ("chuva", "precipitação"),
    ("alegre", "feliz"), ("rápido", "veloz"), ("frio", "gelado"),
    ("casa", "habitação"), ("livro", "leitura"),
]
# Held out in their entirety from association supervision.
TEST_PAIRS = [
    ("banana", "fruta"), ("comboio", "transporte"),
    ("sofá", "móvel"), ("triste", "infeliz"),
    ("bicicleta", "veículo"), ("lua", "satélite"),
]
PT_PT_PRIORITY = ("autocarro", "comboio", "telemóvel", "ficheiro", "ecrã", "equipa", "facto")
PT_BR_SIGNALS = ("ônibus", "celular", "arquivo", "tela", "equipe")
FOREIGN_SIGNALS = ("the", "and", "with", "from")


def encode_bytes(text, bos=False, eos=False):
    return ([1] if bos else []) + [4 + b for b in text.encode("utf-8")] + ([2] if eos else [])


def forced_loss(model, ids, prefix_length=0):
    """Average token negative log likelihood, with targets masked before answer."""
    if len(ids) < 2:
        return float("inf")
    if len(ids) > model.context + 1:
        return float("inf")
    x = torch.tensor(ids[:-1], dtype=torch.long)[None]
    y = torch.tensor(ids[1:], dtype=torch.long)
    y[:max(0, prefix_length - 1)] = -100
    if not (y != -100).any():
        return float("inf")
    return float(F.cross_entropy(model(x)[0], y, ignore_index=-100))


def nonword(word):
    # Transpose neighbouring letters; skip identical/real-word collisions elsewhere.
    for i in range(1, len(word) - 1):
        if word[i] != word[i + 1]:
            return word[:i] + word[i+1] + word[i] + word[i+2:]
    return word[::-1] + "x"


@torch.inference_mode()
def recognition(model, words, count=60):
    model.eval()
    vocabulary = set(words)
    candidates = []
    for word in words:
        fake = nonword(word)
        if fake == word or fake in vocabulary or len(word) < 4:
            continue
        candidates.append((word, fake))
        if len(candidates) >= count:
            break
    records = []
    for word, fake in candidates:
        # Both candidates encoded with the exact same byte tokenizer path.
        a = forced_loss(model, encode_bytes(word, bos=True, eos=True))
        b = forced_loss(model, encode_bytes(fake, bos=True, eos=True))
        records.append({"word": word, "nonword": fake,
                        "real_loss": round(a, 4), "nonword_loss": round(b, 4),
                        "correct": a < b})
    return records


@torch.inference_mode()
def association(model, tok):
    model.eval()
    answers = sorted(set(answer for _, answer in TRAIN_PAIRS + TEST_PAIRS))
    results = []
    for word, expected in TEST_PAIRS:
        prefix = tok.encode(f"Associação: {word} -> ", bos=True)
        # Compare candidate answer likelihoods; no open-ended generation.
        scores = {}
        for answer in answers:
            suffix = tok.encode(answer, eos=True)
            ids = prefix + suffix
            scores[answer] = forced_loss(model, ids, len(prefix))
        predicted = min(scores, key=scores.get)
        results.append({"word": word, "expected": expected, "predicted": predicted,
                        "correct": predicted == expected})
    return results


def corpus_audit(corpus):
    tokens = re.findall(r"[^\W\d_]+", corpus.lower(), re.UNICODE)
    counts = Counter(tokens)
    return {
        "pt_pt_priority": {w: counts[w] for w in PT_PT_PRIORITY},
        "pt_br_signals": {w: counts[w] for w in PT_BR_SIGNALS},
        "foreign_signals": {w: counts[w] for w in FOREIGN_SIGNALS},
        "note": "Heuristic counts, not automatic language classification or corpus filtering.",
    }


def print_vocabulary_report(model, tok, words, corpus):
    results = recognition(model, words)
    associations = association(model, tok)
    def rate(items):
        return round(100 * sum(p["correct"] for p in items) / len(items), 1) if items else None
    print("NORD BRAIN — VOCABULÁRIO (avaliação independente)", flush=True)
    print(json.dumps({"recognition_tests": len(results),
                      "recognition_accuracy_pct": rate(results),
                      "association_holdout_tests": len(associations),
                      "association_accuracy_pct": rate(associations)},
                     ensure_ascii=False), flush=True)
    print("PALAVRAS RECONHECIDAS:", ", ".join(p["word"] for p in results if p["correct"]), flush=True)
    print("PALAVRAS COM DIFICULDADES:", ", ".join(p["word"] for p in results if not p["correct"]), flush=True)
    print("ASSOCIAÇÕES RESERVADAS:", flush=True)
    for item in associations:
        print(json.dumps(item, ensure_ascii=False), flush=True)
    print("AUDITORIA DO CORPUS:", json.dumps(corpus_audit(corpus), ensure_ascii=False), flush=True)
    print("NOTA: reconhecimento = preferência entre bytes; não comprova semântica. "
          "Associação = escolha entre respostas candidatas, não conversa livre.", flush=True)
