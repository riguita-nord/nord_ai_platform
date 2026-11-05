"""Tiny *trainable* contextual intent head owned by Nord Brain, CPU-only.

It does not use a pretrained language model. It is a supervised neural
classifier of the next conversational *action*, not evidence of understanding
Portuguese freely. Its guard abstains when confidence is low, and it never
turns contextual requests into factual teacher lessons.

Feature hashing is a deterministic tokenizer of character/word n-grams.
The trainable parameters live in a local PyTorch checkpoint.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata
import zlib

import torch
from torch import nn
from torch.nn import functional as F

LABELS=('new_topic','ending','longer','chapters','continue',
        'rewrite','explain','discuss')
SIZE=4096
_CACHE: tuple[dict[str,torch.Tensor],float] | None=None
# All examples below are synthetic, author-reviewed demonstrations.
# Holdouts are never passed to the optimizer.
_TRAIN={
'ending':(
    'consegues por com um final melhor',
    'podes melhorar o final da história',
    'o final nao me convence',
    'faz o desfecho mais emocionante',
    'queria outro desfecho',
    'escreve um fim diferente',
    'o final está fraco',
    'preciso de um final mais forte',
    'troca a conclusão da narrativa',
    'faz o fim mais surpreendente',
    'podes dar um novo final',
    'altera o desenlace',
    'gostava de um fim mais bonito',
    'vamos mudar o desfecho',
    'não gostei do final que escreveste',
    'faz um desfecho menos previsível',
    'quero melhorar como a história termina',
    'recria a última cena da história',
    'põe um final mais emocionante',
    'muda o que acontece no fim',
),
'longer':(
    'estica a história',
    'faz essa versão mais extensa',
    'podes acrescentar mais detalhes',
    'desenvolve mais o texto',
    'preciso de mais cenas',
    'faz o conto ainda mais comprido',
    'aumenta o tamanho da resposta',
    'expande a narrativa',
    'gostava de mais desenvolvimento',
    'adiciona mais parágrafos',
    'ainda está demasiado curto',
    'quero uma versão maior',
    'faz mais extenso',
    'consegues fazer maior',
    'dá mais corpo ao enredo',
    'alongar um pouco mais',
    'continua a aumentar o texto',
    'inclui cenas adicionais',
),
'chapters':(
    'divide esta história em capítulos',
    'quero isso com capítulos',
    'organiza a narrativa por capítulos',
    'separa o texto em capítulos',
    'podes transformar isto num livro',
    'faz com capítulos e etc',
    'agora quero vários capítulos',
    'dá uma estrutura de capítulos',
    'põe a história em partes como um romance',
    'com capítulos e títulos',
    'cria capítulos para o conto',
    'faz um livro com esta narrativa',
    'separa em episódios',
    'quero escrever isto por capítulos',
    'adapta o conto para romance',
    'transforma esse conto em livro',
),
'continue':(
    'o que acontece a seguir',
    'escreve a próxima parte',
    'continua a história',
    'não pares agora',
    'quero saber o que acontece depois',
    'avança para o próximo capítulo',
    'prossegue a narrativa',
    'segue daí em diante',
    'e depois o que aconteceu',
    'vamos continuar a história',
    'continua esse enredo',
    'mais um capítulo',
    'diz-me a continuação',
    'o que vem a seguir',
    'escreve a cena seguinte',
    'avança com o que já tínhamos',
),
'rewrite':(
    'escreve de outra forma',
    'faz outra versão da resposta',
    'muda o tom da narrativa',
    'reescreve o texto',
    'podes reformular o que disseste',
    'apresenta de forma mais informal',
    'escreve isso com um estilo diferente',
    'quero mudar a forma de contar',
    'faz uma versão com outro estilo',
    'altera a maneira de explicar',
    'reescreve essa parte',
    'reformula a tua resposta',
    'quero um tom mais sério',
    'transforma isso numa linguagem simples',
    'faz outra abordagem ao mesmo tema',
    'podes reescrever com outras palavras',
),
'explain':(
    'não percebi bem',
    'explica melhor isso',
    'como chegaste a essa conclusão',
    'podes esclarecer a tua ideia',
    'não estou a perceber o teu raciocínio',
    'o que queres dizer com isso',
    'explica o motivo da tua escolha',
    'fala de forma mais clara',
    'preciso que expliques melhor',
    'por que motivo sugeriste isso',
    'podes detalhar essa explicação',
    'não percebo a ligação',
    'explica-me com um exemplo',
    'esclarece esse ponto',
    'não entendi a tua resposta',
    'porquê essa opção',
),
'discuss':(
    'o que achas dessa ideia',
    'e quais são as desvantagens',
    'poderíamos fazer melhor',
    'será que existe outra abordagem',
    'e se isso não funcionar',
    'que alternativas temos',
    'dá a tua opinião sobre isto',
    'compara os pontos fortes e fracos',
    'achas que vale a pena',
    'qual seria a tua escolha',
    'e as vantagens dessa solução',
    'podemos discutir outra possibilidade',
    'como podemos melhorar esta proposta',
    'e quais são os riscos',
    'o que mudarias nessa ideia',
    'terias outra sugestão',
),
'new_topic':(
    'quanto é dois mais dois',
    'quem é o presidente de portugal',
    'o que é python',
    'conta-me uma história',
    'cria um livro sobre robôs',
    'quero aprender matemática',
    'o céu é azul',
    'a capital de portugal é lisboa',
    'corrige a resposta anterior para onze',
    'a resposta correta é quatro',
    'quando nasceu albert einstein',
    'olá como estás',
    'como te chamas',
    'pesquisa sobre a lua',
    'qual é a história de portugal',
    'dois mais dois é quatro',
    'o que sabes sobre astronomia',
    'simula três cenários',
    'quero começar outro assunto',
    'podemos falar agora de carros',
    'a minha cor preferida é verde',
    'isto é uma definição nova',
)
}
_HOLDOUT={
'ending':(
    'consegues pôr a história com um final melhor',
    'podia acabar de maneira mais interessante',
    'vamos fazer a última parte mais surpreendente',
),
'longer':('está pequeno, aumenta um pouco','queria uma resposta mais desenvolvida'),
'chapters':('com vários capítulos por favor','transforma essa história num romance'),
'continue':('continua da parte onde ficaste','então o que vem depois'),
'rewrite':('podes dizer o mesmo de maneira diferente','altera o estilo da tua explicação'),
'explain':('estou confuso com essa parte','ajuda-me a compreender melhor'),
'discuss':('e o que sugerias como alternativa','quais os contras dessa ideia'),
'new_topic':('quanto é dez dividido por dois','ensina-me sobre química','o que é a lua'),
}


def _normalize(text: str) -> str:
    text=unicodedata.normalize('NFKD',str(text).casefold())
    text=''.join(ch for ch in text if not unicodedata.combining(ch))
    return ' '.join(re.findall(r'[a-z0-9]+',text))


def vector(text: str, focus: str='dialogue') -> torch.Tensor:
    """Deterministic lexical features; fully local, no external tokenizer."""
    clean=_normalize(text)[:210]
    words=clean.split()
    feat=[]
    feat.extend('w:'+w for w in words)
    feat.extend('b:'+a+'_'+b for a,b in zip(words,words[1:]))
    padded=' '+clean+' '
    for n in (3,4,5):
        feat.extend('c:'+padded[i:i+n] for i in range(max(0,len(padded)-n+1)))
    feat.append('focus:'+focus)
    tensor=torch.zeros(SIZE,dtype=torch.float32)
    for token in set(feat):
        tensor[zlib.crc32(token.encode('utf-8'))%SIZE]+=1.
    return F.normalize(tensor,p=2,dim=0) if feat else tensor


def _examples(heldout=False):
    source=_HOLDOUT if heldout else _TRAIN
    rows=[]
    for label,samples in source.items():
        # Narrative intentions must also be learned in book context.
        for text in samples:
            focuses=('story','book') if label in (
                'ending','longer','chapters','continue','rewrite') else (
                    ('dialogue','story') if label in ('explain','discuss') else ('dialogue','story','book'))
            for focus in focuses:
                rows.append((text,focus,LABELS.index(label)))
    return rows


def _signature() -> str:
    return hashlib.sha256(repr((_TRAIN,_HOLDOUT,SIZE,LABELS)).encode()).hexdigest()[:24]


class NeuralIntent:
    """Trainable head for context operations, owned by the Nord Brain runtime."""
    def __init__(self,root: Path):
        self.path=Path(root)/'intent-head-v1.pt'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.head=nn.Linear(SIZE,len(LABELS))
        self.validation_accuracy=0.
        if self.path.exists():
            try:
                checkpoint=torch.load(self.path,map_location='cpu',weights_only=True)
                if checkpoint.get('signature')==_signature():
                    self.head.load_state_dict(checkpoint['weights'])
                    self.validation_accuracy=float(checkpoint['heldout_accuracy'])
                    self.head.eval()
                    return
            except (OSError,ValueError,KeyError,RuntimeError):
                pass
        # During tests and multiple Brain instances in one process, reuse
        # the same validated in-memory bootstrap parameters.
        global _CACHE
        if _CACHE is not None:
            self.head.load_state_dict(_CACHE[0])
            self.validation_accuracy=_CACHE[1]
            self.head.eval()
            self._save_checkpoint()
            return
        self.train_bootstrap()

    def _save_checkpoint(self):
        fd,temp=tempfile.mkstemp(prefix='.intent-',suffix='.pt',dir=self.path.parent)
        try:
            with os.fdopen(fd,'wb') as stream:
                torch.save({'signature':_signature(),'weights':self.head.state_dict(),
                            'heldout_accuracy':self.validation_accuracy},stream)
                stream.flush();os.fsync(stream.fileno())
            os.replace(temp,self.path)
        finally:
            if os.path.exists(temp):os.unlink(temp)

    def train_bootstrap(self):
        """One-time supervised fitting; never learns from arbitrary user turns."""
        torch.manual_seed(173)
        self.head=nn.Linear(SIZE,len(LABELS))
        rows=_examples()
        x=torch.stack([vector(q,f) for q,f,_ in rows])
        y=torch.tensor([label for _,_,label in rows],dtype=torch.long)
        optim=torch.optim.AdamW(self.head.parameters(),lr=.08,weight_decay=.012)
        old_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for _ in range(105):
                optim.zero_grad(set_to_none=True)
                logits=self.head(x)
                loss=F.cross_entropy(logits,y)
                loss.backward()
                optim.step()
            self.head.eval()
            tests=_examples(heldout=True)
            with torch.inference_mode():
                tx=torch.stack([vector(q,f) for q,f,_ in tests])
                expected=torch.tensor([label for _,_,label in tests])
                self.validation_accuracy=float((self.head(tx).argmax(-1)==expected).float().mean())
        finally:
            torch.set_num_threads(old_threads)
        global _CACHE
        _CACHE=({k:v.detach().clone() for k,v in self.head.state_dict().items()},
                self.validation_accuracy)
        self._save_checkpoint()

    @torch.inference_mode()
    def predict(self,text: str,focus: str='dialogue') -> tuple[str | None,float]:
        if focus not in ('dialogue','story','book') or not text.strip() or len(text)>210:
            return None,0.
        # On out-of-distribution requests, abstain rather than pollute lessons.
        probabilities=self.head(vector(text,focus)).softmax(-1)
        values,indices=torch.topk(probabilities,2)
        choice=LABELS[int(indices[0])]
        confidence=float(values[0])
        if choice=='new_topic' or confidence < .72 or float(values[0]-values[1]) < .34:
            return None,confidence
        return choice,confidence

    def report(self) -> dict:
        return {'heldout_accuracy':round(self.validation_accuracy,3),
                'train_examples':len(_examples()),
                'heldout_examples':len(_examples(heldout=True)),
                'labels':len(LABELS),
                'path':str(self.path)}
