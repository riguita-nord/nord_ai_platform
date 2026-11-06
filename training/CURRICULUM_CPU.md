# Nord Brain — currículo experimental em CPU

Script: `scripts/train_curriculum.py`. Checkpoint separado: `~/.nord-brain/curriculum/curriculum.pt`. Não modifica os modelos V1, V2 ou V3.

```bash
cd ~/nord_ai_platform
git pull origin main
.venv/bin/python -m compileall -q scripts/train_curriculum.py
NORD_THREADS=2 .venv/bin/python scripts/train_curriculum.py --steps 100 --preview
```

Continuação: `NORD_THREADS=2 .venv/bin/python scripts/train_curriculum.py --steps 500 --preview`. Avaliação sem treino: `NORD_THREADS=2 .venv/bin/python scripts/train_curriculum.py --eval-only`.

Quatro etapas: `words`, `sentences`, `reasoning`, `prediction`. A promoção automática usa limiares provisórios de perda de previsão em pequenos conjuntos reservados, depois de pelo menos 200 atualizações na etapa. **Estes testes não certificam domínio linguístico, raciocínio, previsão ou capacidade de generalização.** O corpus é utilizado nas duas primeiras etapas; nas seguintes são gerados exemplos aritméticos simples com revisão textual a cada cinco atualizações. A fase de previsão é atualmente treino supervisionado, não reinforcement learning.

Limitações conhecidas: tokenizer híbrido de palavras e bytes, não BPE; as etapas de palavras e frases usam atualmente a mesma tarefa de modelação causal de texto; os exemplos reservados são pequenos; não há testes de generalização robustos, sistema de recompensa, RLHF, DPO, PPO ou GRPO. Os limiares poderão nunca ser atingidos. Não aumente parâmetros sem medir RAM/velocidade. Este código ainda não foi executado na VM do utilizador.
