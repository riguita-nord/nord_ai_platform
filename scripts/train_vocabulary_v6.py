"""Nord Brain V6: staged PT-PT vocabulary with stable subword output.

Stages are MANUAL: spelling -> lexical -> paraphrase -> context -> expansion.
Each stage has an independent measured gate; no automatic stage transition.
The final curriculum stage 2 remains locked.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from train_language_v2 import Model, Tokenizer
from train_vocabulary_v2 import TRAIN, TEST

KINDS=("categoria","propriedade","contexto")
STAGES=("spelling","lexical","paraphrase","context","expansion")
CONFIG={"dim":192,"heads":6,"layers":6,"context":256}
Q_TRAIN=("Categoria de {word}: ","{kind} de {word}: ","Indica a {kind} de {word}: ")
Q_HELD="Qual é a {kind} associada a {word}? "
CONTEXTS=(
    "Ficha: {word}; categoria: {cat}; propriedade: {prop}; contexto: {ctx}.\nPergunta: {question}",
    "Informação sobre {word}: categoria={cat}, propriedade={prop}, contexto={ctx}.\nPergunta: {question}",
)
CONTEXT_HELD="Dados: {word} é da categoria {cat}, tem propriedade {prop} e contexto {ctx}.\nPergunta: {question}"
BASE={w:TRAIN[w] for w in list(TRAIN)[:10]}
# Pseudo-concepts are training-only for transfer: held-out real concepts are never optimized.
SYNTH={f"termo{i}":(f"classe{i}",f"ação{i}",f"situação{i}") for i in range(12)}
SYNTH_TEST={f"termo{i}":(f"classe{i}",f"ação{i}",f"situação{i}") for i in range(12,16)}

def question(word,kind,template):
    return template.format(word=word,kind=kind)

def context(word,rels,kind,qt,ct):
    return ct.format(word=word,cat=rels[0],prop=rels[1],ctx=rels[2],
                     question=question(word,kind,qt))

def build_tokenizer():
    # Word tokens reduce byte-by-byte spelling drift for known words;
    # byte fallback still supports unseen strings and PT accents.
    corpus=[]
    for word,rels in {**TRAIN,**SYNTH,**SYNTH_TEST}.items():
        for kind in KINDS:
            for qt in Q_TRAIN+(Q_HELD,):
                corpus.append(question(word,kind,qt))
                for ct in CONTEXTS+(CONTEXT_HELD,):
                    corpus.append(context(word,rels,kind,qt,ct))
        corpus.extend(rels)
    return Tokenizer.fit("\n".join(corpus),limit=2048)

def pair(tok,text,answer,context_limit):
    prefix=tok.encode(text,bos=True)
    seq=prefix+tok.encode(answer,eos=True)
    if len(seq)>context_limit+1:raise ValueError("Sequence exceeds context")
    return seq,len(prefix)

def loss_for(model,seq,prefix):
    x=torch.tensor(seq[:-1],dtype=torch.long)[None]
    y=torch.tensor(seq[1:],dtype=torch.long)
    y[:prefix-1]=-100
    return F.cross_entropy(model(x)[0],y,ignore_index=-100)

@torch.inference_mode()
def generate(model,tok,prompt,max_tokens=24):
    model.eval()
    ids=tok.encode(prompt,bos=True)
    output=[]
    for _ in range(max_tokens):
        logits=model(torch.tensor(ids[-model.context:],dtype=torch.long)[None])[0,-1].clone()
        logits[0]=logits[1]=logits[3]=-float("inf")
        nxt=int(logits.argmax())
        if nxt==2:break
        if nxt<4:break
        output.append(nxt)
        ids.append(nxt)
    return tok.decode(output).strip()

def evaluate(model,tok,concepts,qt,ct=None):
    correct=0
    rows=[]
    for word,rels in concepts.items():
        for idx,kind in enumerate(KINDS):
            p=context(word,rels,kind,qt,ct) if ct else question(word,kind,qt)
            actual=generate(model,tok,p)
            good=actual==rels[idx]
            correct+=good
            rows.append({"word":word,"kind":kind,"expected":rels[idx],"actual":actual,"ok":good})
    return round(100*correct/max(1,len(rows)),1),rows

def report(model,tok,stage,verbose=False):
    scores={}
    for label,items,qt,ct in (
        ("memorized",BASE,Q_TRAIN[0],None),
        ("heldout_paraphrase",BASE,Q_HELD,None),
        ("trained_context",BASE,Q_TRAIN[0],CONTEXTS[0]),
        ("synthetic_context_transfer",SYNTH_TEST,Q_HELD,CONTEXT_HELD),
        ("unseen_context_transfer",TEST,Q_HELD,CONTEXT_HELD),
        ("unseen_no_context",TEST,Q_HELD,None),
        ("expanded_lexical",TRAIN,Q_TRAIN[0],None),
    ):
        value,rows=evaluate(model,tok,items,qt,ct)
        scores[label]=value
        wrong=[r for r in rows if not r["ok"]]
        print(label,":",value,"% |",len(rows)-len(wrong),"/",len(rows),
              "| erros:",json.dumps(wrong if verbose else wrong[:2],ensure_ascii=False),flush=True)
    gates={
        "spelling":scores["memorized"]>=80,
        "lexical":scores["memorized"]>=95,
        "paraphrase":scores["heldout_paraphrase"]>=85 and scores["memorized"]>=90,
        "context":scores["synthetic_context_transfer"]>=80 and scores["trained_context"]>=80,
        "expansion":scores["expanded_lexical"]>=90 and scores["heldout_paraphrase"]>=85,
    }
    print("AVALIAÇÃO V6:",json.dumps({"stage":stage,"scores":scores,
        "current_stage_gate_passed":gates[stage],"stage_2_unlocked":False,
        "meaning_mastery_verified":False},ensure_ascii=False),flush=True)
    return gates[stage]

def examples(tok,stage):
    buckets=[]
    def add(items,templates,contexts=None):
        for word,rels in items.items():
            for idx,kind in enumerate(KINDS):
                for qt in templates:
                    if contexts:
                        for ct in contexts:
                            buckets.append(pair(tok,context(word,rels,kind,qt,ct),rels[idx],CONFIG["context"]))
                    else:
                        buckets.append(pair(tok,question(word,kind,qt),rels[idx],CONFIG["context"]))
    # Rehearsal avoids catastrophic forgetting as complexity increases.
    add(BASE,Q_TRAIN[:1])
    if stage in ("paraphrase","context","expansion"):
        add(BASE,Q_TRAIN[1:])
    if stage in ("context","expansion"):
        add(BASE,Q_TRAIN,CONTEXTS)
        add(SYNTH,Q_TRAIN,CONTEXTS)
    if stage=="expansion":
        add(TRAIN,Q_TRAIN)
        add(TRAIN,Q_TRAIN,CONTEXTS)
    return buckets

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=STAGES,default="spelling")
    ap.add_argument("--steps",type=int,default=500)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--rate",type=float,default=.0002)
    ap.add_argument("--eval-only",action="store_true")
    ap.add_argument("--verbose",action="store_true")
    ap.add_argument("--data-dir",type=Path,default=Path.home()/".nord-brain"/"curriculum")
    args=ap.parse_args()
    if not 1<=args.steps<=100000 or not 1<=args.batch_size<=32 or not 0<args.rate<=.001:
        ap.error("Invalid steps, batch-size or rate")
    torch.set_num_threads(max(1,min(4,int(os.getenv("NORD_THREADS","2")))))
    torch.manual_seed(1729)
    path=args.data_dir/"vocabulary-v6.pt"
    if path.exists():
        state=torch.load(path,map_location="cpu",weights_only=True)
        tok=Tokenizer(state["tokenizer"]["words"])
        model=Model(len(tok.vocab),**state["config"])
        model.load_state_dict(state["model"])
        optim=torch.optim.AdamW(model.parameters(),lr=args.rate)
        optim.load_state_dict(state["optimizer"])
        for group in optim.param_groups:group["lr"]=args.rate
        rng=random.Random()
        rng.setstate(state["rng"])
        steps=state["steps"]
        unlocked=state["unlocked"]
        print("A retomar V6:",steps,"etapas aprovadas:",unlocked,flush=True)
    else:
        if args.eval_only:ap.error("Checkpoint V6 inexistente")
        tok=build_tokenizer()
        model=Model(len(tok.vocab),**CONFIG)
        optim=torch.optim.AdamW(model.parameters(),lr=args.rate)
        rng=random.Random(1729)
        steps=0
        unlocked=[]
        print("Novo V6:",len(tok.vocab),"tokens,",sum(p.numel() for p in model.parameters()),"parâmetros",flush=True)
    index=STAGES.index(args.stage)
    if index and STAGES[index-1] not in unlocked:
        ap.error("Etapa bloqueada: aprova primeiro "+STAGES[index-1])
    if args.eval_only:
        report(model,tok,args.stage,args.verbose)
        return
    data=examples(tok,args.stage)
    for _ in range(args.steps):
        model.train()
        optim.zero_grad(set_to_none=True)
        chosen=rng.sample(data,k=min(args.batch_size,len(data)))
        loss=torch.stack([loss_for(model,seq,prefix) for seq,prefix in chosen]).mean()
        if not torch.isfinite(loss):raise RuntimeError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        steps+=1
        if steps%100==0:print(json.dumps({"steps":steps,"loss":round(float(loss.detach()),4)}),flush=True)
    passed=report(model,tok,args.stage,args.verbose)
    if passed and args.stage not in unlocked:unlocked.append(args.stage)
    args.data_dir.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    torch.save({"config":CONFIG,"tokenizer":tok.to_dict(),"model":model.state_dict(),
                "optimizer":optim.state_dict(),"rng":rng.getstate(),
                "steps":steps,"unlocked":unlocked},tmp)
    os.replace(tmp,path)
    print("Checkpoint:",path,"| etapa aprovada:",passed,flush=True)
if __name__=="__main__":main()
