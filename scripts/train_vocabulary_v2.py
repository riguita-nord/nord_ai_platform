"""Nord Brain vocabulary V2: PT-PT concept learning with an explicit promotion gate.

This experiment does not modify curriculum-four.pt. It is not a claim of
human-like semantic understanding. Usage: --steps 500 / --eval-only.
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
from train_language_v2 import Model
from vocabulary_eval import encode_bytes

# Curated PT-PT concept families. Each concept has multiple distinct relations.
# Entire concepts are reserved for evaluation; training cannot memorise them.
CONCEPTS = {
    "água": ("líquido", "beber", "sede"),
    "leite": ("líquido", "beber", "alimentação"),
    "cão": ("animal", "doméstico", "companhia"),
    "gato": ("animal", "doméstico", "companhia"),
    "maçã": ("fruta", "comer", "alimentação"),
    "pera": ("fruta", "comer", "alimentação"),
    "banana": ("fruta", "comer", "alimentação"),
    "carro": ("veículo", "conduzir", "transporte"),
    "autocarro": ("veículo", "passageiros", "transporte"),
    "comboio": ("veículo", "passageiros", "transporte"),
    "bicicleta": ("veículo", "pedalar", "transporte"),
    "mesa": ("móvel", "casa", "apoio"),
    "cadeira": ("móvel", "casa", "sentar"),
    "sofá": ("móvel", "casa", "sentar"),
    "sol": ("astro", "luz", "céu"),
    "lua": ("astro", "noite", "céu"),
    "chuva": ("meteorologia", "água", "céu"),
    "neve": ("meteorologia", "frio", "céu"),
    "livro": ("objeto", "ler", "conhecimento"),
    "caderno": ("objeto", "escrever", "escola"),
    "telemóvel": ("tecnologia", "comunicação", "chamadas"),
    "computador": ("tecnologia", "trabalho", "informação"),
    "médico": ("profissão", "saúde", "hospital"),
    "professor": ("profissão", "ensino", "escola"),
    "cozinheiro": ("profissão", "comida", "cozinha"),
    "alegre": ("emoção", "feliz", "positivo"),
    "triste": ("emoção", "infeliz", "negativo"),
    "rápido": ("velocidade", "veloz", "movimento"),
    "lento": ("velocidade", "devagar", "movimento"),
    "quente": ("temperatura", "calor", "sensação"),
    "frio": ("temperatura", "gelado", "sensação"),
    "equipa": ("grupo", "pessoas", "cooperação"),
    "família": ("grupo", "pessoas", "parentesco"),
    "ficheiro": ("informática", "dados", "computador"),
    "ecrã": ("tecnologia", "imagem", "visualização"),
}
# Hold out entire concepts, not merely rewordings of trained examples.
HELD_OUT = {"banana", "comboio", "sofá", "lua", "caderno", "professor", "triste", "lento", "frio", "família"}
TRAIN = {k:v for k,v in CONCEPTS.items() if k not in HELD_OUT}
TEST = {k:v for k,v in CONCEPTS.items() if k in HELD_OUT}
ANSWERS = sorted({v[0] for v in CONCEPTS.values()})
PROMPTS = ("Categoria de {word}: ", "Classe de {word}: ", "Tipo de {word}: ")
# This benchmark tests the first relation only; other relations are taught
# to increase supervision but cannot prove full word meaning.
def example(word, relation, prompt):
    prefix = encode_bytes(prompt.format(word=word), bos=True)
    return prefix + encode_bytes(relation, eos=True), len(prefix)


@torch.inference_mode()
def score(model, word, answer, template, adjust_prior=True):
    """Conditional log probability of complete answer, adjusted by unconditional prior.

    Unlike mean token loss, this does not reward short answers simply for
    having fewer target tokens. Prior adjustment reduces label popularity bias.
    """
    def log_likelihood(prompt_word):
        ids, prefix = example(prompt_word, answer, template)
        x = torch.tensor(ids[:-1], dtype=torch.long)[None]
        targets = torch.tensor(ids[1:], dtype=torch.long)
        logits = model(x)[0]
        token_logp = F.log_softmax(logits, dim=-1)
        positions = torch.arange(prefix - 1, len(targets))
        return float(token_logp[positions, targets[positions]].sum())
    conditional = log_likelihood(word)
    return conditional - log_likelihood("") if adjust_prior else conditional




@torch.inference_mode()
def evaluate(model, items, template=None, adjust_prior=True):
    model.eval()
    rows = []
    template = template or PROMPTS[0]
    for word, relations in items.items():
        # Fixed question format; only category labels are valid candidates.
        scores = {answer: score(model, word, answer, template, adjust_prior) for answer in ANSWERS}
        predicted = min(scores, key=scores.get)
        rows.append({"word":word, "expected":relations[0],
                     "predicted":predicted, "correct":predicted == relations[0]})
    return rows


def report(model, diagnostic=False):
    train = evaluate(model, TRAIN)
    test = evaluate(model, TEST)
    def accuracy(rows):
        return sum(r["correct"] for r in rows) / len(rows)
    train_acc, test_acc = accuracy(train), accuracy(test)
    # Diagnostic gate only; the shared language model is not unlocked by this script.
    passed = train_acc >= .90 and test_acc >= .80
    print(json.dumps({"train_concepts":len(train), "held_out_concepts":len(test),
                      "train_accuracy_pct":round(100*train_acc,1),
                      "generalisation_accuracy_pct":round(100*test_acc,1),
                      "vocabulary_probe_passed":passed, "stage_2_unlocked":False}, ensure_ascii=False),flush=True)
    print("PALAVRAS DE TREINO:",flush=True)
    for row in train: print(json.dumps(row,ensure_ascii=False),flush=True)
    print("CONCEITOS RESERVADOS:",flush=True)
    for row in test: print(json.dumps(row,ensure_ascii=False),flush=True)
    if diagnostic:
        for template in (PROMPTS[0], "Relação de {word}: "):
            for prior in (False, True):
                tr = evaluate(model, TRAIN, template, prior)
                te = evaluate(model, TEST, template, prior)
                print(json.dumps({"diagnostic_prompt":template, "prior_adjustment":prior,
                    "train_accuracy_pct":round(100*accuracy(tr),1),
                    "heldout_accuracy_pct":round(100*accuracy(te),1),
                    "predicted_distribution":dict(__import__("collections").Counter(row["predicted"] for row in tr))},
                    ensure_ascii=False),flush=True)
    print("NOTA: teste de classificação por categoria, com ajuste ao prior; "
          "não prova que o modelo saiba o significado de todas as palavras.",flush=True)
    return passed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps",type=int,default=500)
    ap.add_argument("--eval-only",action="store_true")
    ap.add_argument("--diagnose",action="store_true",help="Compare original/new prompts and raw/prior-adjusted scores")
    ap.add_argument("--rate",type=float,default=.0002)
    ap.add_argument("--data-dir",type=Path,default=Path.home()/".nord-brain"/"curriculum")
    args=ap.parse_args()
    if not 1<=args.steps<=100000 or not 0<args.rate<=.001: ap.error("Invalid steps/rate")
    torch.set_num_threads(max(1,min(4,int(os.getenv("NORD_THREADS","2")))))
    torch.manual_seed(1729)
    rng=random.Random(1729)
    path=args.data_dir/"vocabulary-v2.pt"
    if path.exists():
        state=torch.load(path,map_location="cpu",weights_only=True)
        model=Model(260,**state["config"])
        model.load_state_dict(state["model"])
        opt=torch.optim.AdamW(model.parameters(),lr=args.rate)
        opt.load_state_dict(state["optimizer"])
        for g in opt.param_groups:g["lr"]=args.rate
        done=state["steps"]
        print("A retomar V2:",done,flush=True)
    else:
        if args.eval_only:ap.error("No V2 checkpoint found; run training first")
        config={"dim":192,"heads":6,"layers":6,"context":256}
        model=Model(260,**config)
        opt=torch.optim.AdamW(model.parameters(),lr=args.rate)
        done=0
        print("Novo modelo de vocabulário V2:",sum(p.numel() for p in model.parameters()),flush=True)
    if args.eval_only:
        report(model, diagnostic=args.diagnose)
        return
    # One unambiguous category target per question; other relations require separately labelled tasks.
    examples=[example(w,rels[0],p) for w,rels in TRAIN.items() for p in PROMPTS]
    for i in range(args.steps):
        ids,prefix=rng.choice(examples)
        x=torch.tensor(ids[:-1],dtype=torch.long)[None]
        y=torch.tensor(ids[1:],dtype=torch.long)
        y[:prefix-1]=-100
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x)[0],y,ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        done+=1
        if done%100==0:
            print(json.dumps({"steps":done,"loss":round(loss.item(),4)}),flush=True)
    args.data_dir.mkdir(parents=True,exist_ok=True)
    state={"config":{"dim":192,"heads":6,"layers":6,"context":256},
           "model":model.state_dict(),"optimizer":opt.state_dict(),"steps":done}
    tmp=path.with_suffix(".tmp")
    torch.save(state,tmp)
    os.replace(tmp,path)
    print("Checkpoint:",path,flush=True)
    report(model, diagnostic=args.diagnose)


if __name__=="__main__":
    main()
