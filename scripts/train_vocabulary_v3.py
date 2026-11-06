"""Nord Brain vocabulary V3: measurable neural learning, no automatic phase promotion.

Separate checkpoint from V1/V2. First prove a 10-concept overfit, then expand.
Uses one causal Transformer, byte tokens and supervised answer-only loss.
No rules are used to produce model answers; the lexicon supplies training labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from train_language_v2 import Model
from vocabulary_eval import encode_bytes
from train_vocabulary_v2 import TRAIN, TEST

# Each type of relationship has a distinct prompt and target.
QUESTIONS = {
    "category": ("Categoria de {word}: ", 0),
    "property": ("Relação de {word}: ", 1),
    "context": ("Contexto de {word}: ", 2),
}
# Held-out phrasings evaluate whether the model generalises beyond one template.
PARAPHRASES = {
    "category": "Que categoria tem {word}? ",
    "property": "Que relação tem {word}? ",
    "context": "Qual o contexto de {word}? ",
}
DIAGNOSTIC_WORDS = tuple(list(TRAIN)[:10])
CONFIG = {"dim":192,"heads":6,"layers":6,"context":256}


def sample(word, relations, kind, template=None):
    prompt, idx = QUESTIONS[kind]
    prefix=encode_bytes((template or prompt).format(word=word),bos=True)
    answer=relations[idx]
    seq=prefix+encode_bytes(answer,eos=True)
    if len(seq)>CONFIG["context"]+1:
        raise ValueError("Training example exceeds context")
    return seq,len(prefix),answer


def loss_for(model, seq, prefix):
    x=torch.tensor(seq[:-1],dtype=torch.long)[None]
    y=torch.tensor(seq[1:],dtype=torch.long)
    y[:prefix-1]=-100
    return F.cross_entropy(model(x)[0],y,ignore_index=-100)


@torch.inference_mode()
def generate(model, prompt, limit=40):
    model.eval()
    ids=encode_bytes(prompt,bos=True)
    out=[]
    for _ in range(limit):
        logits=model(torch.tensor(ids[-model.context:],dtype=torch.long)[None])[0,-1].clone()
        logits[0]=logits[1]=logits[3]=-float("inf")
        next_id=int(logits.argmax())
        if next_id==2:break
        if next_id<4:break
        out.append(next_id-4)
        ids.append(next_id)
    return bytes(out).decode("utf-8",errors="replace").strip()


@torch.inference_mode()
def evaluate(model, concepts, paraphrase=False):
    rows=[]
    for word, relations in concepts.items():
        for kind,(template,idx) in QUESTIONS.items():
            prompt=(PARAPHRASES[kind] if paraphrase else template).format(word=word)
            predicted=generate(model,prompt)
            expected=relations[idx]
            rows.append({"word":word,"kind":kind,"expected":expected,
                         "predicted":predicted,"correct":predicted==expected})
    return rows


def report(model, active, mode):
    seen=evaluate(model,active)
    paraphrases=evaluate(model,active,True)
    unseen=evaluate(model,TEST)
    def accuracy(rows):return round(100*sum(x["correct"] for x in rows)/len(rows),1) if rows else 0.
    metrics={"mode":mode,"trained_concepts":len(active),"training_accuracy_pct":accuracy(seen),
             "paraphrase_accuracy_pct":accuracy(paraphrases),
             "unseen_concepts":len(TEST),"unseen_accuracy_pct":accuracy(unseen),
             "stage_2_unlocked":False}
    print("AVALIAÇÃO V3:",json.dumps(metrics,ensure_ascii=False),flush=True)
    print("DISTRIBUIÇÃO DAS RESPOSTAS:",json.dumps(
        Counter(row["predicted"] for row in seen).most_common(12),ensure_ascii=False),flush=True)
    for label,rows in (("TREINO",seen),("REFORMULAÇÕES",paraphrases),("CONCEITOS NÃO TREINADOS",unseen)):
        print(label,flush=True)
        for row in rows:print(json.dumps(row,ensure_ascii=False),flush=True)
    if mode=="diagnostic":
        print("TESTE 10 PALAVRAS:", "APROVADO" if accuracy(seen)>=95 else "NÃO APROVADO",flush=True)
    print("AVISO: os conceitos não treinados não têm informação semântica garantida. "
          "Esta avaliação não comprova compreensão geral da língua.",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",choices=("diagnostic","full"),default="diagnostic")
    ap.add_argument("--steps",type=int,default=500)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--rate",type=float,default=.0002)
    ap.add_argument("--eval-only",action="store_true")
    ap.add_argument("--data-dir",type=Path,default=Path.home()/".nord-brain"/"curriculum")
    args=ap.parse_args()
    if not 1<=args.steps<=100000 or not 1<=args.batch_size<=64 or not 0<args.rate<=.001:
        ap.error("Invalid steps/batch-size/rate")
    torch.set_num_threads(max(1,min(4,int(os.getenv("NORD_THREADS","2")))))
    torch.manual_seed(1729)
    path=args.data_dir/"vocabulary-v3.pt"
    active={w:TRAIN[w] for w in DIAGNOSTIC_WORDS} if args.mode=="diagnostic" else TRAIN
    if path.exists():
        state=torch.load(path,map_location="cpu",weights_only=True)
        model=Model(260,**state["config"])
        model.load_state_dict(state["model"])
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.rate)
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:group["lr"]=args.rate
        steps=state["steps"]
        rng=random.Random()
        rng.setstate(state["rng_state"])
        print("A retomar V3:",steps,flush=True)
        if args.mode=="full" and not state.get("diagnostic_passed",False):
            ap.error("Full mode blocked: diagnostic must reach >=95% exact training answers first")
        passed=state.get("diagnostic_passed",False)
    else:
        if args.eval_only:ap.error("V3 checkpoint not found")
        if args.mode!="diagnostic":ap.error("Run --mode diagnostic before full")
        model=Model(260,**CONFIG)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.rate)
        rng=random.Random(1729)
        steps=0
        passed=False
        print("Novo cérebro V3:",sum(p.numel() for p in model.parameters()),"parâmetros",flush=True)
    if args.eval_only:
        report(model,active,args.mode)
        return
    examples=[sample(word,relations,kind)[:2] for word,relations in active.items() for kind in QUESTIONS]
    for i in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # True gradient accumulation across a balanced cycle, not a single label per update.
        choices=rng.sample(examples,k=min(args.batch_size,len(examples)))
        total=torch.stack([loss_for(model,seq,prefix) for seq,prefix in choices]).mean()
        if not torch.isfinite(total):raise RuntimeError("Non-finite loss")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        steps+=1
        if steps%100==0:
            print(json.dumps({"steps":steps,"loss":round(float(total.detach()),4)}),flush=True)
    if args.mode=="diagnostic":
        seen=evaluate(model,active)
        passed=passed or sum(row["correct"] for row in seen)/len(seen)>=.95
    args.data_dir.mkdir(parents=True,exist_ok=True)
    state={"config":CONFIG,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
           "steps":steps,"rng_state":rng.getstate(),"diagnostic_passed":passed}
    tmp=path.with_suffix(".tmp")
    torch.save(state,tmp)
    os.replace(tmp,path)
    print("Checkpoint:",path,flush=True)
    report(model,active,args.mode)


if __name__=="__main__":
    main()
