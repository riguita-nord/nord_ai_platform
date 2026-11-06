"""Nord Brain — laboratório de cinco palavras (PT-PT).

Treino supervisionado real, sem respostas hardcoded no motor de inferência.
Não promove checkpoints automaticamente. Requer PyTorch e train_language_v2.py.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from train_language_v2 import Model, Tokenizer

WORDS = {
    "água": {"classe": "substantivo", "genero": "feminino", "tipo": "líquido", "plural": "águas"},
    "bola": {"classe": "substantivo", "genero": "feminino", "tipo": "objeto", "plural": "bolas"},
    "vaca": {"classe": "substantivo", "genero": "feminino", "tipo": "animal", "plural": "vacas"},
    "carro": {"classe": "substantivo", "genero": "masculino", "tipo": "veículo", "plural": "carros"},
    "andar": {"classe": "verbo", "genero": "não se aplica", "tipo": "ação", "plural": "não se aplica"},
}
CONJUGATION = {"eu": "ando", "tu": "andas", "ele": "anda", "nós": "andamos", "vós": "andais", "eles": "andam"}
TEMPLATES_TRAIN = {
    "classe": ["Qual é a classe gramatical de {w}?", "A palavra {w} é um substantivo ou um verbo?"],
    "genero": ["Qual é o género gramatical de {w}?"],
    "tipo": ["O que é {w}?", "A que categoria pertence {w}?"],
    "plural": ["Qual é o plural de {w}?"],
}
TEMPLATES_TEST = {
    "classe": ["Como classificas gramaticalmente a palavra {w}?"],
    "genero": ["A palavra {w} é masculina ou feminina?"],
    "tipo": ["Diz-me a categoria de {w}."],
    "plural": ["Como se escreve {w} no plural?"],
}
CONFIG = {"dim": 192, "layers": 6, "heads": 6, "context": 256}

def examples(templates):
    pairs = []
    for word, facts in WORDS.items():
        for key, variants in templates.items():
            for template in variants:
                pairs.append((template.format(w=word), facts[key]))
    return pairs

def conjugation_examples(heldout=False):
    template = "Como se conjuga andar com {p}?" if heldout else "Conjuga o verbo andar para {p}."
    return [(template.format(p=person), form) for person, form in CONJUGATION.items()]

def sentence_examples(heldout=False):
    if heldout:
        return [
            ("Completa: A vaca ____ no campo.", "anda"),
            ("Completa: O carro ____ na estrada.", "anda"),
            ("Qual é o verbo na frase 'A vaca anda'?", "anda"),
            ("Qual é o substantivo na frase 'A vaca anda'?", "vaca"),
        ]
    return [
        ("Na frase 'A vaca anda', qual é a ação?", "anda"),
        ("Na frase 'O carro anda', qual é a ação?", "anda"),
        ("Completa com o verbo andar: Eu ____.", "ando"),
        ("Completa com o verbo andar: Tu ____.", "andas"),
    ]

def prompt(question):
    return "Pessoa: " + question + "\nNord: "

def supervised(tok, question, answer):
    prefix = tok.encode(prompt(question), bos=True)
    ids = prefix + tok.encode(answer, eos=True)
    if len(ids) > CONFIG["context"] + 1:
        raise ValueError("Exemplo excede o contexto")
    return ids, len(prefix)

def loss_for(model, pair):
    ids, start = pair
    x = torch.tensor(ids[:-1], dtype=torch.long)[None]
    y = torch.tensor(ids[1:], dtype=torch.long)
    y[:start - 1] = -100
    return F.cross_entropy(model(x)[0], y, ignore_index=-100)

@torch.inference_mode()
def answer(model, tok, question, max_tokens=48):
    model.eval()
    ids = tok.encode(prompt(question), bos=True)
    if len(ids) >= CONFIG["context"]:
        return ""
    output = []
    for _ in range(max_tokens):
        logits = model(torch.tensor(ids[-CONFIG["context"]:], dtype=torch.long)[None])[0, -1].clone()
        logits[0] = logits[1] = logits[3] = -float("inf")
        if not output:
            logits[2] = -float("inf")
        nxt = int(logits.argmax())
        if nxt == 2:
            break
        output.append(nxt)
        ids.append(nxt)
    return tok.decode(output).split("\nPessoa:")[0].strip()

def norm(value):
    return value.casefold().strip().rstrip(".!? ")

def evaluate(model, tok, tests):
    correct = 0
    mistakes = []
    for question, expected in tests:
        actual = answer(model, tok, question)
        ok = norm(actual) == norm(expected)
        correct += int(ok)
        if not ok and len(mistakes) < 8:
            mistakes.append({"pergunta": question, "esperado": expected, "obtido": actual})
    return {"acertos": correct, "total": len(tests),
            "precisao_pct": round(100 * correct / max(1, len(tests)), 1),
            "erros": mistakes}

def save(path, model, opt, tok, step, rng, best):
    state = {"config": CONFIG, "model": model.state_dict(),
             "optimizer": opt.state_dict(), "tokenizer": tok.to_dict(),
             "step": step, "rng": rng.getstate(), "best": best,
             "experiment": "five-words-v1"}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--rate", type=float, default=0.0002)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=Path.home()/".nord-brain"/"five-words")
    args = ap.parse_args()
    if not 1 <= args.steps <= 100000 or not 1 <= args.batch_size <= 16:
        ap.error("Passos/batch inválidos")
    if not 1 <= args.eval_every <= 10000 or not 0 < args.rate <= 0.001:
        ap.error("Intervalo/taxa inválidos")
    torch.set_num_threads(max(1, min(4, int(os.getenv("NORD_THREADS", "2")))))
    rng = random.Random(1729)
    torch.manual_seed(1729)
    train = examples(TEMPLATES_TRAIN) + conjugation_examples() + sentence_examples()
    test = examples(TEMPLATES_TEST) + conjugation_examples(True) + sentence_examples(True)
    # Tokenizer is built from training material only; unknown words use UTF-8 bytes.
    text = "\n".join(prompt(q) + a for q, a in train)
    tok = Tokenizer.fit(text, limit=512)
    model = Model(len(tok.vocab), **CONFIG)
    opt = torch.optim.AdamW(model.parameters(), lr=args.rate)
    path = args.data_dir/"five-words-v1.pt"
    best_path = args.data_dir/"five-words-v1-best.pt"
    step, best = 0, -1
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("experiment") != "five-words-v1" or state["tokenizer"] != tok.to_dict():
            ap.error("Checkpoint incompatível")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        for group in opt.param_groups:
            group["lr"] = args.rate
        step, best = state["step"], state["best"]
        rng.setstate(state["rng"])
        print("A retomar no passo", step, flush=True)
    elif args.eval_only:
        ap.error("Ainda não existe checkpoint")
    if args.eval_only:
        print(json.dumps(evaluate(model, tok, test), ensure_ascii=False), flush=True)
        return
    encoded = [supervised(tok, q, a) for q, a in train]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"modelo": "Nord Brain 5 Palavras", "palavras": list(WORDS),
                      "exemplos_treino": len(train), "perguntas_reservadas": len(test),
                      "tokens": len(tok.vocab), "parametros": sum(p.numel() for p in model.parameters())},
                     ensure_ascii=False), flush=True)
    for i in range(args.steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = torch.stack([loss_for(model, rng.choice(encoded)) for _ in range(args.batch_size)]).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Loss não finita")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(json.dumps({"passo": step, "loss": round(float(loss.detach()), 4)}), flush=True)
        if step % args.eval_every == 0 or i == args.steps - 1:
            result = evaluate(model, tok, test)
            print("AVALIAÇÃO CINCO PALAVRAS:", json.dumps({"passo": step, **result}, ensure_ascii=False), flush=True)
            if result["precisao_pct"] > best:
                best = result["precisao_pct"]
                save(best_path, model, opt, tok, step, rng, best)
            save(path, model, opt, tok, step, rng, best)
    print("Treino concluído. Checkpoint:", path,
          "| aprovação de produção: NÃO | generalização para palavras novas: NÃO TESTADA", flush=True)

if __name__ == "__main__":
    main()
