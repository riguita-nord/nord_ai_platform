"""Treinador único offline para o NordTransformer original (NeuralCore).

Pré-treino de texto PT-PT, sem outro Transformer ou tokenizer. Usa o mesmo
checkpoint que a aplicação; preserva cópia anterior e valida antes de guardar.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from nord_brain.core import NeuralCore, BOS

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path.home()/".nord-brain"/"brain-v2.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--rate", type=float, default=.00008)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--max-regression", type=float, default=.03)
    args = ap.parse_args()
    if not 0 <= args.steps <= 100000 or not 0 < args.rate <= .001 or not 0 <= args.max_regression <= 1:
        ap.error("Parâmetros inválidos")
    if not args.corpus.is_file():
        ap.error("Corpus não encontrado")
    data = args.corpus.read_bytes()
    if len(data) < 100000:
        ap.error("Corpus insuficiente: mínimo 100 KB")
    data.decode("utf-8")
    torch.set_num_threads(max(1,min(4,int(os.getenv("NORD_THREADS","2")))))
    core = NeuralCore(args.checkpoint)
    model = core.model
    context = model.config.context
    # Separação de avaliação no final do corpus; não usar validação para updates.
    cut = int(len(data)*.95)
    train, valid = data[:cut], data[cut:]
    if len(train) < context+2 or len(valid) < context+2:
        ap.error("Corpus insuficiente para janela de contexto")
    rng = random.Random(1729 + core.steps)
    def loss_at(buf, offset):
        span = buf[offset:offset+context]
        ids = torch.tensor([BOS]+list(span[:-1]),dtype=torch.long)[None]
        targets = torch.tensor(list(span),dtype=torch.long)
        return F.cross_entropy(model(ids)[0],targets)
    @torch.inference_mode()
    def validation():
        model.eval()
        stops = [int(i*(len(valid)-context)/7) for i in range(8)]
        return sum(float(loss_at(valid,s)) for s in stops)/len(stops)
    before = validation()
    print(json.dumps({"modelo":"NordTransformer principal","passos_anteriores":core.steps,
                      "loss_validacao_inicial":round(before,4),
                      "checkpoint":str(args.checkpoint)},ensure_ascii=False),flush=True)
    if args.eval_only or args.steps == 0:
        return
    # Nunca sobrescrever o único checkpoint anterior sem backup.
    backup = args.checkpoint.with_suffix(".pre-unified.pt")
    if args.checkpoint.exists() and not backup.exists():
        shutil.copy2(args.checkpoint,backup)
    for group in core.optimizer.param_groups:
        group["lr"] = args.rate
    for i in range(args.steps):
        model.train()
        offset = rng.randrange(len(train)-context)
        core.optimizer.zero_grad(set_to_none=True)
        loss = loss_at(train,offset)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss não finita; checkpoint original mantido")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        core.optimizer.step()
        core.steps += 1
        if (i+1)%100 == 0 or i+1 == args.steps:
            print(json.dumps({"passo":i+1,"loss_treino":round(float(loss.detach()),4),
                              "loss_validacao":round(validation(),4)},ensure_ascii=False),flush=True)
    after = validation()
    if after > before + args.max_regression:
        print(json.dumps({"aceite":False,"antes":round(before,4),
                          "depois":round(after,4),"motivo":"regressão na validação; pesos não guardados"},
                         ensure_ascii=False),flush=True)
        return
    core.save()
    print(json.dumps({"aceite":True,"antes":round(before,4),"depois":round(after,4),
                      "passos_totais":core.steps,"checkpoint":str(args.checkpoint)},
                     ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()
