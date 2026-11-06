"""Nord Brain V5: lexical, paraphrase and in-context transfer experiments.

A single trainable causal Transformer, not a rules-based answer engine.
Training uses 25 concepts; ten different concepts are never gradient-trained.
The transfer test supplies factual context for unseen concepts in the prompt.
Checkpoint is isolated; stage 2 never unlocks automatically.
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

CONFIG={"dim":192,"layers":6,"heads":6,"context":256}
KINDS=("categoria","propriedade","contexto")
# These templates are training data, not inference-time answer rules.
TRAIN_QUESTIONS=(
    "Qual é a {kind} de {word}? ",
    "Indica a {kind} de {word}: ",
    "Sobre {word}, qual é a {kind}? ",
    "Diz a {kind} de {word}: ",
)
HELDOUT_QUESTION="Consegues identificar a {kind} associada a {word}? "
CONTEXT_TEMPLATES=(
    "Informação: {word} tem categoria {cat}, propriedade {prop} e contexto {ctx}.\nPergunta: {question}",
    "Ficha de {word}: categoria={cat}; propriedade={prop}; contexto={ctx}.\nPergunta: {question}",
)
HELDOUT_CONTEXT="Dados sobre {word}: a categoria é {cat}; a propriedade é {prop}; o contexto é {ctx}.\nPergunta: {question}"


def prompt(word,relations,kind,question_template,context_template=None):
    question=question_template.format(word=word,kind=kind)
    if context_template is None:return question
    return context_template.format(word=word,cat=relations[0],prop=relations[1],
                                   ctx=relations[2],question=question)


def example(text,answer):
    prefix=encode_bytes(text,bos=True)
    seq=prefix+encode_bytes(answer,eos=True)
    if len(seq)>CONFIG["context"]+1:
        raise ValueError("Example exceeds model context; reduce prompt length")
    return seq,len(prefix)


def loss_for(model,seq,prefix):
    x=torch.tensor(seq[:-1],dtype=torch.long)[None]
    labels=torch.tensor(seq[1:],dtype=torch.long)
    labels[:prefix-1]=-100
    return F.cross_entropy(model(x)[0],labels,ignore_index=-100)


@torch.inference_mode()
def generate(model,text,max_bytes=48):
    model.eval()
    ids=encode_bytes(text,bos=True)
    output=[]
    for _ in range(max_bytes):
        logits=model(torch.tensor(ids[-model.context:],dtype=torch.long)[None])[0,-1].clone()
        logits[0]=logits[1]=logits[3]=-float("inf")
        token=int(logits.argmax())
        if token==2:break
        if token<4:break
        output.append(token-4)
        ids.append(token)
    return bytes(output).decode("utf-8",errors="replace").strip()


@torch.inference_mode()
def evaluate(model,concepts,question_template,context_template=None):
    rows=[]
    for word,relations in concepts.items():
        for index,kind in enumerate(KINDS):
            text=prompt(word,relations,kind,question_template,context_template)
            predicted=generate(model,text)
            rows.append({"word":word,"relation":kind,"expected":relations[index],
                         "predicted":predicted,"correct":predicted==relations[index]})
    return rows


def accuracy(rows):
    return round(100*sum(row["correct"] for row in rows)/len(rows),1) if rows else 0.


def report(model,verbose=False):
    # Four distinct measures: memorisation, linguistic paraphrase,
    # unsupported zero-shot guessing, and factual transfer with context.
    metrics={}
    tests={
        "trained_questions":evaluate(model,TRAIN,TRAIN_QUESTIONS[0]),
        "heldout_paraphrases":evaluate(model,TRAIN,HELDOUT_QUESTION),
        "unseen_without_context":evaluate(model,TEST,HELDOUT_QUESTION),
        "unseen_with_context":evaluate(model,TEST,HELDOUT_QUESTION,HELDOUT_CONTEXT),
        "trained_with_context":evaluate(model,TRAIN,TRAIN_QUESTIONS[0],CONTEXT_TEMPLATES[0]),
    }
    for name,rows in tests.items():metrics[name+"_accuracy_pct"]=accuracy(rows)
    metrics.update({"trained_concepts":len(TRAIN),"unseen_concepts":len(TEST),
                    "stage_2_unlocked":False,"meaning_mastery_verified":False})
    print("AVALIAÇÃO V5:",json.dumps(metrics,ensure_ascii=False),flush=True)
    for name,rows in tests.items():
        wrong=[row for row in rows if not row["correct"]]
        print(name,":",len(rows)-len(wrong),"/",len(rows),
              "erros:",json.dumps(wrong if verbose else wrong[:5],ensure_ascii=False),flush=True)
    print("NOTA: contexto de teste contém factos explícitos; acertos medem transferência "
          "a partir desses factos, não conhecimento prévio nem compreensão geral.",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps",type=int,default=500)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--rate",type=float,default=.0002)
    ap.add_argument("--eval-only",action="store_true")
    ap.add_argument("--verbose",action="store_true")
    ap.add_argument("--data-dir",type=Path,default=Path.home()/".nord-brain"/"curriculum")
    args=ap.parse_args()
    if not 1<=args.steps<=100000 or not 1<=args.batch_size<=32 or not 0<args.rate<=.001:
        ap.error("Invalid steps, batch size or rate")
    torch.set_num_threads(max(1,min(4,int(os.getenv("NORD_THREADS","2")))))
    torch.manual_seed(1729)
    path=args.data_dir/"vocabulary-v5.pt"
    if path.exists():
        state=torch.load(path,map_location="cpu",weights_only=True)
        model=Model(260,**state["config"])
        model.load_state_dict(state["model"])
        optim=torch.optim.AdamW(model.parameters(),lr=args.rate)
        optim.load_state_dict(state["optimizer"])
        for group in optim.param_groups:group["lr"]=args.rate
        rng=random.Random()
        rng.setstate(state["rng_state"])
        steps=state["steps"]
        print("A retomar V5:",steps,flush=True)
    else:
        if args.eval_only:ap.error("V5 checkpoint not found")
        model=Model(260,**CONFIG)
        optim=torch.optim.AdamW(model.parameters(),lr=args.rate)
        rng=random.Random(1729)
        steps=0
        print("Novo modelo V5:",sum(p.numel() for p in model.parameters()),"parâmetros",flush=True)
    if args.eval_only:
        report(model,args.verbose)
        return
    # All training answers originate from TRAIN, never from TEST.
    # Interleave direct associations with context reading; no inference-time lookup.
    buckets={"direct":[],"context":[]}
    for word,relations in TRAIN.items():
        for index,kind in enumerate(KINDS):
            for question in TRAIN_QUESTIONS:
                buckets["direct"].append(example(prompt(word,relations,kind,question),relations[index]))
                for context in CONTEXT_TEMPLATES:
                    buckets["context"].append(example(
                        prompt(word,relations,kind,question,context),relations[index]))
    for _ in range(args.steps):
        model.train()
        optim.zero_grad(set_to_none=True)
        # Balanced task selection per update; avoid the same examples repeatedly.
        losses=[]
        for j in range(args.batch_size):
            bucket=buckets["direct" if j%2==0 else "context"]
            seq,prefix=rng.choice(bucket)
            losses.append(loss_for(model,seq,prefix))
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise RuntimeError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        steps+=1
        if steps%100==0:
            print(json.dumps({"steps":steps,"loss":round(float(loss.detach()),4)}),flush=True)
    args.data_dir.mkdir(parents=True,exist_ok=True)
    state={"config":CONFIG,"model":model.state_dict(),"optimizer":optim.state_dict(),
           "rng_state":rng.getstate(),"steps":steps}
    temp=path.with_suffix(".tmp")
    torch.save(state,temp)
    os.replace(temp,path)
    print("Checkpoint:",path,flush=True)
    report(model,args.verbose)


if __name__=="__main__":
    main()
