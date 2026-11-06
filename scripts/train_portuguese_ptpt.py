"""Nord Brain PT-PT: corpus-grounded linguistic training with isolated evaluation.

CPU-first experimental trainer. No pretrained weights or external model APIs.
The checkpoint is NOT promoted automatically to production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from train_language_v2 import Model, Tokenizer

CONFIG = {"dim": 192, "heads": 6, "layers": 6, "context": 256}
DIALOGUE = re.compile(r"(?m)^Diálogo:\\s*$")
TURN = re.compile(r"^(Pessoa|Nord):\\s*(.*)$")
STAGES = ("foundation", "dialogue", "comprehension")
# Independent composition benchmark: novel names/objects are reserved from training.
NAMES_TRAIN = ("Ana", "João", "Maria", "Pedro", "Rita", "Luís")
NAMES_TEST = ("Inês", "Tiago", "Beatriz", "Miguel")
OBJECTS_TRAIN = ("livro", "caderno", "casaco", "telemóvel", "jornal")
OBJECTS_TEST = ("mapa", "relógio", "chapéu", "dicionário")
VERBS = (("comprou", "comprou"), ("encontrou", "encontrou"), ("guardou", "guardou"))
QUESTIONS = (("Quem {v} o {o}?", 0), ("O que é que {n} {v}?", 1))
TRAIN_PREFIX = "Pessoa: "
ANSWER_PREFIX = "\\nNord: "

def read_corpus(path):
    text = path.read_text(encoding="utf-8")
    if len(text.strip()) < 1000:
        raise ValueError("Corpus demasiado pequeno")
    return text

def parse_dialogues(text):
    pairs = []
    for block in DIALOGUE.split(text)[1:]:
        turns = []
        for line in block.splitlines():
            match = TURN.match(line.strip())
            if match:
                turns.append((match.group(1), match.group(2)))
            elif turns and line.strip() and not line.startswith("==="):
                # Ignore free-form annotations outside a turn.
                continue
        for i in range(1, len(turns)):
            if turns[i-1][0] == "Pessoa" and turns[i][0] == "Nord":
                pairs.append((turns[i-1][1], turns[i][1]))
    return sorted(set(pairs))

def composition(names, objects):
    examples = []
    for name in names:
        for obj in objects:
            for verb, _ in VERBS:
                statement = f"{name} {verb} o {obj}."
                for template, target in QUESTIONS:
                    question = template.format(n=name, v=verb, o=obj)
                    answer = name if target == 0 else f"o {obj}"
                    examples.append((f"Texto: {statement}\\nPergunta: {question}", answer))
    return examples

def make_prompt(question):
    return TRAIN_PREFIX + question + ANSWER_PREFIX

def encode(tok, prompt, answer, limit):
    prefix = tok.encode(prompt, bos=True)
    ids = prefix + tok.encode(answer, eos=True)
    if len(ids) > limit + 1:
        return None
    return ids, len(prefix)

def loss_for(model, pair):
    ids, prefix = pair
    x = torch.tensor(ids[:-1], dtype=torch.long)[None]
    labels = torch.tensor(ids[1:], dtype=torch.long)
    labels[:prefix-1] = -100
    return F.cross_entropy(model(x)[0], labels, ignore_index=-100)

@torch.inference_mode()
def generate(model, tok, question, max_tokens=60):
    model.eval()
    ids = tok.encode(make_prompt(question), bos=True)
    if len(ids) >= model.context:
        return ""
    result = []
    for _ in range(max_tokens):
        logits = model(torch.tensor(ids[-model.context:], dtype=torch.long)[None])[0, -1].clone()
        logits[0] = logits[1] = logits[3] = -float("inf")
        if not result:
            logits[2] = -float("inf")
        next_id = int(logits.argmax())
        if next_id == 2:
            break
        result.append(next_id)
        ids.append(next_id)
    return tok.decode(result).split("\\nPessoa:")[0].strip()

def score(model, tok, pairs):
    correct = 0
    examples = []
    for question, expected in pairs:
        actual = generate(model, tok, question)
        correct += actual.casefold().rstrip(".!? ") == expected.casefold().rstrip(".!? ")
        if len(examples) < 4 and actual.casefold().rstrip(".!? ") != expected.casefold().rstrip(".!? "):
            examples.append({"question": question, "expected": expected, "actual": actual})
    return {"correct": correct, "total": len(pairs),
            "accuracy_pct": round(100 * correct / max(1, len(pairs)), 1),
            "errors": examples}

def evaluate(model, tok, dialogue_test, stage):
    result = {
        "dialogue_holdout": score(model, tok, dialogue_test),
        "composition_holdout": score(model, tok, composition(NAMES_TEST, OBJECTS_TEST)[:48]),
    }
    result["promotion_eligible"] = (
        result["dialogue_holdout"]["accuracy_pct"] >= 90
        and result["composition_holdout"]["accuracy_pct"] >= 90
    )
    result["stage"] = stage
    result["phase_2_unlocked"] = False
    return result

def save(path, model, opt, rng, stage, step, best, corpus_hash, words):
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
             "config": CONFIG, "tokenizer": {"words": words},
             "rng": rng.getstate(), "stage": stage, "step": step,
             "best": best, "corpus_sha256": corpus_hash}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=STAGES, default="foundation")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rate", type=float, default=0.0002)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--corpus", type=Path, default=Path("training/nord_brain_treino_pt_pt.txt"))
    ap.add_argument("--data-dir", type=Path, default=Path.home()/".nord-brain"/"pt-pt")
    args = ap.parse_args()
    if not 1 <= args.steps <= 100000 or not 1 <= args.batch_size <= 16:
        ap.error("Passos ou batch inválidos")
    if not 0 < args.rate <= 0.001 or not 1 <= args.eval_every <= 10000:
        ap.error("Taxa ou intervalo inválidos")
    torch.set_num_threads(max(1, min(4, int(os.getenv("NORD_THREADS", "2")))))
    torch.manual_seed(1729)
    rng = random.Random(1729)
    corpus = read_corpus(args.corpus)
    digest = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    dialogues = parse_dialogues(corpus)
    if len(dialogues) < 10:
        ap.error("Corpus precisa de pelo menos 10 diálogos")
    # Hold out entire dialogue pairs; no gradient updates on these examples.
    rng.shuffle(dialogues)
    hold_count = max(3, len(dialogues) // 5)
    held_dialogue = dialogues[:hold_count]
    train_dialogue = dialogues[hold_count:]
    train_composition = composition(NAMES_TRAIN, OBJECTS_TRAIN)
    tokenizer_text = corpus + "\\n" + "\\n".join(
        q + " " + a for q, a in train_composition
    )
    tok = Tokenizer.fit(tokenizer_text, limit=1024)
    path = args.data_dir/"language-ptpt-v1.pt"
    best_path = args.data_dir/"language-ptpt-v1-best.pt"
    model = Model(len(tok.vocab), **CONFIG)
    opt = torch.optim.AdamW(model.parameters(), lr=args.rate)
    step, best = 0, -1.
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["corpus_sha256"] != digest or state["tokenizer"]["words"] != tok.to_dict()["words"]:
            ap.error("Corpus/tokenizador mudou; utiliza outro --data-dir para novo treino")
        if state["stage"] != args.stage:
            ap.error("Etapa diferente do checkpoint; usa outro --data-dir (sem promoção automática)")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        for group in opt.param_groups:
            group["lr"] = args.rate
        rng.setstate(state["rng"])
        step, best = state["step"], state["best"]
        print("A retomar treino PT-PT:", step, flush=True)
    elif args.eval_only:
        ap.error("Checkpoint inexistente")
    if args.eval_only:
        print(json.dumps(evaluate(model, tok, held_dialogue, args.stage), ensure_ascii=False), flush=True)
        return
    print("Corpus PT-PT:", len(corpus), "caracteres | diálogos treino:",
          len(train_dialogue), "| reservados:", len(held_dialogue), flush=True)
    language_tokens = tok.encode(corpus)
    # Text-only corpus is used for language modeling; held-out dialogue pairs
    # occur in that corpus. Therefore dialogue_holdout is a diagnostic only,
    # NOT a leakage-free generalisation metric. Composition holdout is independent.
    if len(language_tokens) < CONFIG["context"] + 1:
        ap.error("Corpus insuficiente")
    supervised = []
    for q, a in train_dialogue + train_composition:
        pair = encode(tok, make_prompt(q), a, CONFIG["context"])
        if pair is not None:
            supervised.append(pair)
    if not supervised:
        ap.error("Sem exemplos supervisionados")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.batch_size):
            if args.stage == "foundation" and rng.random() < 0.65:
                start = rng.randrange(len(language_tokens) - CONFIG["context"])
                seq = language_tokens[start:start+CONFIG["context"]+1]
                x = torch.tensor(seq[:-1], dtype=torch.long)[None]
                y = torch.tensor(seq[1:], dtype=torch.long)
                losses.append(F.cross_entropy(model(x)[0], y))
            else:
                losses.append(loss_for(model, rng.choice(supervised)))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Loss não finita")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": round(float(loss.detach()), 4)}), flush=True)
        if step % args.eval_every == 0 or i == args.steps - 1:
            metrics = evaluate(model, tok, held_dialogue, args.stage)
            print("AVALIAÇÃO PT-PT:", json.dumps(metrics, ensure_ascii=False), flush=True)
            # Only the leakage-free composition score selects best checkpoint.
            value = metrics["composition_holdout"]["accuracy_pct"]
            if value > best:
                best = value
                save(best_path, model, opt, rng, args.stage, step, best, digest, tok.to_dict()["words"])
            save(path, model, opt, rng, args.stage, step, best, digest, tok.to_dict()["words"])
    print("Checkpoint:", path, "| fase 2 bloqueada | aprovação manual necessária", flush=True)

if __name__ == "__main__":
    main()
