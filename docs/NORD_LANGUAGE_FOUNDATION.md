# Nord Language Foundation — experiência PT-PT

Esta experiência usa o Transformer próprio do projeto (6 camadas, 192 dimensões).
Não usa modelos externos nem altera o modelo de produção.

## Dados

O pré-treino requer um ficheiro UTF-8 de texto português europeu, com pelo
menos 100 KB. Por omissão: `training/dataset/train.txt`.

Verifica manualmente a qualidade, licença e origem do corpus. Um corpus pequeno
ou repetitivo pode produzir perda baixa sem boa generalização. O script divide
os tokens em 95% treino / 5% validação por posição, não mistura exemplos;
documentos duplicados entre secções podem contaminar a avaliação. Para uma
avaliação rigorosa, prepara documentos separados e testes externos.

## 1. Pré-treinar

```bash
cd ~/nord_ai_platform
git pull origin main
.venv/bin/python -m compileall -q scripts/train_foundation_ptpt.py scripts/train_five_words.py
NORD_THREADS=2 .venv/bin/python scripts/train_foundation_ptpt.py \
  --corpus training/dataset/train.txt \
  --steps 2000 \
  --data-dir ~/.nord-brain/foundation-ptpt-v1
```

O comando pode ser repetido para continuar mais 2000 passos. Os pesos ficam
em `~/.nord-brain/foundation-ptpt-v1/foundation-ptpt-v1.pt`.
A validação reporta perda, **não** percentagem de compreensão.

## 2. Especializar nas cinco palavras

```bash
NORD_THREADS=2 .venv/bin/python scripts/train_five_words.py \
  --foundation ~/.nord-brain/foundation-ptpt-v1/foundation-ptpt-v1.pt \
  --fresh --data-dir ~/.nord-brain/five-words-from-foundation-v1 \
  --cycles 12 --steps-per-cycle 250 --diagnose-every 50 \
  --batch-size 4 --patience 5 --target 100
```

Para retomar o fine-tuning, repete sem `--fresh`, com o mesmo
`--foundation` e `--data-dir`.

## 3. Avaliar

```bash
NORD_THREADS=2 .venv/bin/python scripts/train_five_words.py \
  --data-dir ~/.nord-brain/five-words-from-foundation-v1 --eval-only

NORD_THREADS=2 .venv/bin/python scripts/train_five_words.py \
  --data-dir ~/.nord-brain/five-words-from-foundation-v1 --final-test
```

O teste final deve ser usado com parcimónia, não para escolher
hiperparâmetros. Os 20 exemplos de validação são reformulações sobre as
mesmas cinco palavras; não provam generalização a novas palavras.

## Limitações

Esta é uma primeira integração de pré-treino + fine-tuning, **não** uma base
linguística validada. Ainda falta avaliar um tokenizer subword alternativo,
construir um corpus PT-PT de qualidade comprovada, testes semânticos com
palavras inéditas e proteção contra esquecimento catastrófico.
Não promover automaticamente para produção.
