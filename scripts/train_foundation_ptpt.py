"""Pré-treino causal PT-PT para Nord Brain. Sem modelos externos.
O corpus deve ser texto PT-PT licenciado/consentido e limpo pelo utilizador.
Checkpoint separado; nunca substitui automaticamente o modelo de produção.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from train_language_v2 import Model, Tokenizer

CONFIG = {"dim": 192, "layers": 6, "heads": 6, "context": 256}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("training/dataset/train.txt"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--rate", type=float, default=0.0002)
    ap.add_argument("--data-dir", type=Path, default=Path.home()/".nord-brain"/"foundation-ptpt")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.steps <= 1000000 or not 512 <= args.vocab <= 8192 or not 0 < args.rate <= .001:
        ap.error("Parâmetros inválidos")
    if not args.corpus.is_file():
        ap.error("Corpus inexistente: " + str(args.corpus))
    raw = args.corpus.read_bytes()
    if len(raw) < 100000:
        ap.error("Corpus demasiado pequeno: mínimo 100 KB de texto PT-PT")
    corpus = raw.decode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    torch.set_num_threads(max(1, min(4, int(os.getenv("NORD_THREADS", "2")))))
    rng = random.Random(1729)
    torch.manual_seed(1729)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    path = args.data_dir/"foundation-ptpt-v1.pt"
    if args.fresh and path.exists():
        ap.error("--fresh exige diretório sem checkpoint; não apaga dados")
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("experiment") != "foundation-ptpt-v1" or state["corpus_sha256"] != digest:
            ap.error("Checkpoint incompatível com o corpus ou versão")
        tok = Tokenizer(state["tokenizer"]["words"])
        model = Model(len(tok.vocab), **state["config"])
        model.load_state_dict(state["model"])
        opt = torch.optim.AdamW(model.parameters(), lr=args.rate)
        opt.load_state_dict(state["optimizer"])
        for group in opt.param_groups:
            group["lr"] = args.rate
        rng.setstate(state["rng"])
        step = state["step"]
    else:
        if args.eval_only:
            ap.error("Checkpoint ainda não existe")
        tok = Tokenizer.fit(corpus, limit=args.vocab)
        model = Model(len(tok.vocab), **CONFIG)
        opt = torch.optim.AdamW(model.parameters(), lr=args.rate)
        step = 0
    ids = tok.encode(corpus)
    # Split contíguo: avaliação é a cauda do corpus, nunca usada no treino.
    cut = int(len(ids) * .95)
    train, heldout = ids[:cut], ids[cut:]
    if len(train) <= CONFIG["context"] + 1 or len(heldout) <= CONFIG["context"] + 1:
        ap.error("Corpus tokenizado insuficiente")
    def batch_loss(data, start):
        seq = torch.tensor(data[start:start+CONFIG["context"]+1], dtype=torch.long)
        return F.cross_entropy(model(seq[:-1][None])[0], seq[1:])
    @torch.inference_mode()
    def validation():
        model.eval()
        starts = [int(i*(len(heldout)-CONFIG["context"]-1)/7) for i in range(8)]
        return round(sum(float(batch_loss(heldout, i)) for i in starts)/len(starts), 4)
    print(json.dumps({"modelo":"Nord Foundation PT-PT v1", "tokens":len(ids),
          "vocabulario":len(tok.vocab),"passo_inicial":step,
          "validacao_loss":validation()},ensure_ascii=False),flush=True)
    if args.eval_only:
        return
    for i in range(args.steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        start = rng.randrange(len(train)-CONFIG["context"]-1)
        loss = batch_loss(train, start)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss não finita")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        step += 1
        if (i+1)%100 == 0 or i+1 == args.steps:
            metric = validation()
            state = {"experiment":"foundation-ptpt-v1","config":CONFIG,
                     "tokenizer":tok.to_dict(),"model":model.state_dict(),
                     "optimizer":opt.state_dict(),"rng":rng.getstate(),
                     "step":step,"corpus_sha256":digest,"validation_loss":metric}
            tmp = path.with_suffix(".tmp")
            torch.save(state,tmp)
            os.replace(tmp,path)
            print(json.dumps({"passo":step,"loss_treino":round(float(loss.detach()),4),
                  "loss_validacao":metric},ensure_ascii=False),flush=True)
    print("Checkpoint:",path,"| não promovido para produção",flush=True)

if __name__ == "__main__":
    main()
