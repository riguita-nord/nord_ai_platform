# Nord Brain — treino unificado no Transformer original

O único Transformer oficial é `nord_brain/core.py` (`NordTransformer` / `NeuralCore`).
A aplicação continua a usar os seus comandos de treino internos, que atuam
sobre este mesmo modelo. O novo comando de terminal `scripts/train_nord.py`
permite pré-treinar texto PT-PT no MESMO formato de checkpoint.

## Segurança antes de começar

**Fecha `./nord` e quaisquer outros processos que treinem o cérebro.**
O treinador externo não partilha o bloqueio de escrita da aplicação.
Faz backup de `~/.nord-brain/brain-v2.pt` antes da primeira execução.
O script cria também `brain-v2.pre-unified.pt` se ainda não existir.

## Executar

```bash
cd ~/nord_ai_platform
git pull origin main
.venv/bin/python -m compileall -q scripts/train_nord.py
NORD_THREADS=2 .venv/bin/python scripts/train_nord.py \
  --corpus training/dataset/train.txt \
  --checkpoint ~/.nord-brain/brain-v2.pt \
  --steps 1000
```

`--eval-only` calcula a perda de validação sem atualizar pesos.
O treinador guarda o checkpoint apenas se a perda na secção reservada
não piorar mais do que `--max-regression`. Isto não certifica compreensão
e não substitui testes independentes de gramática e diálogo.

## Arquivos experimentais removidos do Git

Foram retirados `scripts/train_language_v2.py`,
`scripts/train_foundation_ptpt.py` e `scripts/train_five_words.py`.
Os checkpoints experimentais existentes no disco **não são apagados**;
os seus pesos não são compatíveis com o Transformer principal e não
devem ser carregados como `brain-v2.pt`.

Os módulos de currículos `nord_brain/linguistic_training.py`,
`portuguese_training.py` e `dialogue_training.py` continuam a ser
fontes de exercícios e rotinas da aplicação, não Transformers separados.

## Próximas verificações

Precisamos de testes de integração do CLI, avaliação de esquecimento,
generalização com palavras inéditas e prevenção de sobreposição de
documentos entre treino e validação. Não executar este treino em paralelo
com a aplicação.
