"""Small supervised neural classifier for ambiguous everyday turns.

Uses preceding assistant utterance as input. This is an experimental
contextual intent classifier, not a generative language model.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import tempfile
import os
import unicodedata
import zlib

import torch
from torch import nn
from torch.nn import functional as F

LABELS=('answer','question','acknowledge')
DIM=2048

# Separate training and held-out conversations; never learn from generated replies.
TRAIN=(
 ('Como estás?', 'Tudo bem', 'answer'),
 ('Como te sentes?', 'Estou bem', 'answer'),
 ('E tu, como estás?', 'Bem, obrigado', 'answer'),
 ('Está tudo bem contigo?', 'Tudo bem', 'answer'),
 ('Como vai isso?', 'Tudo tranquilo', 'answer'),
 ('Como correu o dia?', 'Bem', 'answer'),
 ('E por aí, tudo bem?', 'Tudo ótimo', 'answer'),
 ('Como tens passado?', 'Estou bem', 'answer'),
 ('Olá!', 'Tudo bem', 'question'),
 ('Boas!', 'Tudo bem', 'question'),
 ('Olá, sou a Nord.', 'Tudo bem', 'question'),
 ('Ora viva!', 'Tudo bem', 'question'),
 ('Bom dia!', 'Tudo bem', 'question'),
 ('Olá de novo!', 'Como estás', 'question'),
 ('Hey!', 'Está tudo bem', 'question'),
 ('Olá, estou por aqui.', 'Tudo bem', 'question'),
 ('Já terminei o trabalho.', 'Tudo bem', 'acknowledge'),
 ('O ficheiro está guardado.', 'Tudo bem', 'acknowledge'),
 ('Podemos começar amanhã.', 'Está bem', 'acknowledge'),
 ('A tarefa ficou concluída.', 'Tudo bem', 'acknowledge'),
 ('O servidor já está pronto.', 'Ok, tudo bem', 'acknowledge'),
 ('Enviei a mensagem.', 'Está bem', 'acknowledge'),
 ('O plano foi atualizado.', 'Tudo bem', 'acknowledge'),
 ('Ficamos por aqui.', 'Está bem', 'acknowledge'),
)
HOLDOUT=(
 ('Como estás hoje?', 'Tudo bem', 'answer'),
 ('E contigo, está tudo bem?', 'Estou bem', 'answer'),
 ('Olá, em que posso ajudar?', 'Tudo bem', 'question'),
 ('Boa tarde!', 'Tudo bem', 'question'),
 ('Já configurei a aplicação.', 'Tudo bem', 'acknowledge'),
 ('A alteração ficou aplicada.', 'Está bem', 'acknowledge'),
)

def _norm(value):
    value=unicodedata.normalize('NFKD',value.casefold())
    return ' '.join(re.findall(r'[a-z0-9]+',''.join(c for c in value if not unicodedata.combining(c))))

def _vector(previous, current):
    features=[]
    for tag,raw in (('p',previous),('u',current)):
        words=_norm(raw).split()
        features.extend(tag+'w:'+w for w in words)
        features.extend(tag+'b:'+a+'_'+b for a,b in zip(words,words[1:]))
        padded=' '+_norm(raw)+' '
        for n in (3,4):
            features.extend(tag+'c:'+padded[i:i+n] for i in range(max(0,len(padded)-n+1)))
        features.append(tag+'question:'+str('?' in raw))
    x=torch.zeros(DIM)
    for feature in set(features):
        x[zlib.crc32(feature.encode())%DIM]+=1
    return F.normalize(x,p=2,dim=0)

class ContextIntent:
    def __init__(self,root:Path):
        self.path=Path(root)/'context-intent-v1.pt'
        signature=hashlib.sha256(repr((TRAIN,HOLDOUT,DIM)).encode()).hexdigest()
        self.model=nn.Linear(DIM,len(LABELS))
        if self.path.exists():
            try:
                data=torch.load(self.path,map_location='cpu',weights_only=True)
                if data['signature']==signature:
                    self.model.load_state_dict(data['weights'])
                    self.model.eval()
                    return
            except (KeyError,RuntimeError,ValueError,OSError):
                pass
        torch.manual_seed(371)
        self.model=nn.Linear(DIM,len(LABELS))
        x=torch.stack([_vector(a,b) for a,b,_ in TRAIN])
        y=torch.tensor([LABELS.index(c) for _,_,c in TRAIN])
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=.06)
        old=torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for _ in range(120):
                optimizer.zero_grad()
                loss=F.cross_entropy(self.model(x),y)
                loss.backward()
                optimizer.step()
        finally:
            torch.set_num_threads(old)
        self.model.eval()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.context-',suffix='.pt',dir=self.path.parent)
        try:
            with os.fdopen(fd,'wb') as stream:
                torch.save({'signature':signature,'weights':self.model.state_dict()},stream)
            os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    @torch.inference_mode()
    def predict(self,previous:str,current:str):
        if not previous or not current or len(current)>80:
            return None,0.
        probs=self.model(_vector(previous,current)).softmax(-1)
        values,indices=torch.topk(probs,2)
        confidence=float(values[0])
        if confidence<.70 or float(values[0]-values[1])<.35:
            return None,confidence
        return LABELS[int(indices[0])],confidence

    @torch.inference_mode()
    def evaluate(self):
        correct=sum(self.predict(a,b)[0]==c for a,b,c in HOLDOUT)
        return {'correct':correct,'total':len(HOLDOUT)}
