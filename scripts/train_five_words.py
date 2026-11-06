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
    "classe": ["Qual é a classe gramatical de {w}?", "A palavra {w} é um substantivo ou um verbo?", "Identifica a classe de {w}."],
    "genero": ["Qual é o género gramatical de {w}?", "Indica o género da palavra {w}.", "A palavra {w} tem que género?"],
    "tipo": ["O que é {w}?", "A que categoria pertence {w}?", "Indica o tipo de {w}."],
    "plural": ["Qual é o plural de {w}?", "Escreve {w} no plural."],
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

def training_diagnostics(model, tok, groups):
    """Diagnóstico APENAS com perguntas de treino; nunca usa validação para ensinar."""
    scores = {}
    for name, pairs in groups.items():
        scores[name] = evaluate(model, tok, pairs)["precisao_pct"]
    return scores


def weighted_training_groups():
    groups = {}
    for key in TEMPLATES_TRAIN:
        groups[key] = [(template.format(w=word), facts[key])
                       for word, facts in WORDS.items()
                       for template in TEMPLATES_TRAIN[key]]
    groups["conjugacao"] = conjugation_examples()
    groups["frases"] = sentence_examples()
    return groups


def choose_group(rng, groups, diagnostic):
    # Reforço baseado exclusivamente em desempenho sobre material de treino.
    # Todas as competências mantêm probabilidade positiva (revisão espaçada simples).
    keys = list(groups)
    weights = [1.0 + (100.0 - diagnostic.get(k, 0.0)) / 20.0 for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def check_dataset(train, validation, final_test):
    sets = [set(q for q, _ in pairs) for pairs in (train, validation, final_test)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Perguntas sobrepostas entre treino, validação e teste final")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=12, help="Número máximo de ciclos")
    ap.add_argument("--steps-per-cycle", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rate", type=float, default=0.0002)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--target", type=float, default=100.0)
    ap.add_argument("--diagnose-every", type=int, default=50)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--final-test", action="store_true", help="Executa teste reservado uma única vez, sem treinar")
    ap.add_argument("--foundation", type=Path, help="Checkpoint pré-treinado PT-PT, usado só ao iniciar experiência nova")
    ap.add_argument("--fresh", action="store_true", help="Inicia experiência nova num diretório vazio")
    ap.add_argument("--data-dir", type=Path, default=Path.home()/".nord-brain"/"five-words-loop")
    args = ap.parse_args()
    if not 1 <= args.cycles <= 1000 or not 1 <= args.steps_per_cycle <= 10000:
        ap.error("Ciclos/passos inválidos")
    if not 1 <= args.batch_size <= 16 or not 1 <= args.patience <= 100 or not 0 < args.rate <= .001:
        ap.error("Parâmetros inválidos")
    if not 1 <= args.diagnose_every <= args.steps_per_cycle:
        ap.error("Diagnóstico inválido")
    if not 0 < args.target <= 100:
        ap.error("Objetivo inválido")
    torch.set_num_threads(max(1, min(4, int(os.getenv("NORD_THREADS", "2")))))
    rng = random.Random(1729)
    torch.manual_seed(1729)
    train = examples(TEMPLATES_TRAIN) + conjugation_examples() + sentence_examples()
    # Validação usa reformulações distintas, mas não deve ser considerada prova
    # independente de compreensão; a avaliação final fica isolada.
    validation = examples(TEMPLATES_TEST)
    final_test = conjugation_examples(True) + sentence_examples(True)
    check_dataset(train, validation, final_test)
    groups = weighted_training_groups()
    text = "\n".join(prompt(q) + a for q, a in train)
    foundation = None
    if args.foundation:
        if not args.foundation.is_file():
            ap.error("Checkpoint foundation inexistente")
        foundation = torch.load(args.foundation, map_location="cpu", weights_only=True)
        if foundation.get("experiment") != "foundation-ptpt-v1" or foundation.get("config") != CONFIG:
            ap.error("Foundation incompatível")
        tok = Tokenizer(foundation["tokenizer"]["words"])
    else:
        tok = Tokenizer.fit(text, limit=512)
    model = Model(len(tok.vocab), **CONFIG)
    opt = torch.optim.AdamW(model.parameters(), lr=args.rate)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    path = args.data_dir/"five-words-loop-v2.pt"
    best_path = args.data_dir/"five-words-loop-v2-best.pt"
    if args.fresh and (path.exists() or best_path.exists()):
        ap.error("--fresh exige --data-dir novo e vazio; não apaga checkpoints")
    cycle, best, stalled = 0, -1.0, 0
    if not path.exists() and foundation is not None and not (args.eval_only or args.final_test):
        model.load_state_dict(foundation["model"])
        print("FOUNDATION:", json.dumps({"passos_pre_treino":foundation["step"],
              "loss_validacao":foundation.get("validation_loss"),
              "modo":"fine-tuning cinco palavras"}, ensure_ascii=False), flush=True)
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not args.foundation:
            tok = Tokenizer(state["tokenizer"]["words"])
            model = Model(len(tok.vocab), **CONFIG)
            opt = torch.optim.AdamW(model.parameters(), lr=args.rate)
        if state.get("experiment") != "five-words-loop-v2" or state["tokenizer"] != tok.to_dict():
            ap.error("Checkpoint incompatível")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"])
        cycle, best, stalled = state["cycle"], state["best"], state["stalled"]
        for group in opt.param_groups:
            group["lr"] = args.rate
        print("A retomar ciclo", cycle, flush=True)
    elif args.eval_only or args.final_test:
        ap.error("Ainda não existe checkpoint")
    if args.eval_only or args.final_test:
        if best_path.exists():
            state = torch.load(best_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
        selected = final_test if args.final_test else validation
        print(json.dumps({"tipo": "teste_final" if args.final_test else "validacao",
                          **evaluate(model, tok, selected)}, ensure_ascii=False), flush=True)
        return
    encoded_groups = {key: [supervised(tok, q, a) for q, a in pairs]
                      for key, pairs in groups.items()}
    diagnostic = {key: 0.0 for key in groups}
    print(json.dumps({"modelo": "Nord Brain 5 Palavras — loop v2",
                      "treino": len(train), "validacao": len(validation),
                      "teste_final_reservado": len(final_test),
                      "tokens": len(tok.vocab)}, ensure_ascii=False), flush=True)
    def snapshot(destination):
        state = {"experiment": "five-words-loop-v2", "config": CONFIG,
                 "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "tokenizer": tok.to_dict(), "rng": rng.getstate(),
                 "cycle": cycle, "best": best, "stalled": stalled}
        temp = destination.with_suffix(".tmp")
        torch.save(state, temp)
        os.replace(temp, destination)
    for _ in range(args.cycles):
        cycle += 1
        for i in range(args.steps_per_cycle):
            if i % args.diagnose_every == 0:
                diagnostic = training_diagnostics(model, tok, groups)
                print("PROFESSOR:", json.dumps({"ciclo": cycle, "passo": i,
                      "dominio_treino_pct": diagnostic,
                      "prioridade": min(diagnostic, key=diagnostic.get)},
                      ensure_ascii=False), flush=True)
            group = choose_group(rng, encoded_groups, diagnostic)
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = torch.stack([loss_for(model, rng.choice(encoded_groups[group])) for _ in range(args.batch_size)]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Loss não finita")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        metrics = evaluate(model, tok, validation)
        diagnostic = training_diagnostics(model, tok, groups)
        current = metrics["precisao_pct"]
        improved = current > best
        if improved:
            best, stalled = current, 0
            snapshot(best_path)
        else:
            stalled += 1
            if best_path.exists():
                saved = torch.load(best_path, map_location="cpu", weights_only=True)
                model.load_state_dict(saved["model"])
                opt.load_state_dict(saved["optimizer"])
                # Reduz o passo de aprendizagem após rollback.
                for group in opt.param_groups:
                    group["lr"] = max(group["lr"] * .7, .00001)
        snapshot(path)
        print("CICLO:", json.dumps({"ciclo": cycle, "loss": round(float(loss.detach()), 4),
              "validacao": metrics, "dominio_treino_pct": diagnostic, "melhor_pct": best, "melhorou": improved,
              "rollback": not improved, "falhas_consecutivas": stalled},
              ensure_ascii=False), flush=True)
        if best >= args.target:
            print("Objetivo de validação atingido; teste final continua separado.", flush=True)
            break
        if stalled >= args.patience:
            print("Paragem por falta de progresso; o melhor snapshot foi preservado.", flush=True)
            break
    print("Melhor snapshot:", best_path,
          "| produção: NÃO | palavras novas: NÃO TESTADAS", flush=True)

if __name__ == "__main__":
    main()
