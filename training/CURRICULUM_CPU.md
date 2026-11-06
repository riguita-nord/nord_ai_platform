# Nord Brain — quatro treinos manuais (CPU)

Cada treino é executado separadamente e apresenta o resultado no fim. **Não existe avanço automático**.

```bash
cd ~/nord_ai_platform
git pull origin main
.venv/bin/python -m compileall -q scripts/train_curriculum.py

# 1. Vocabulário: no fim mostra palavras e a respetiva perda neural
NORD_THREADS=2 .venv/bin/python scripts/train_curriculum.py --stage words --steps 100

# 2. Construção de frases: no fim mostra exemplos gerados
NORD_THREADS=2 .venv/bin/python scripts/train_curriculum.py --stage sentences --steps 100

# 3. Raciocínio e lógica: no fim testa contas não usadas no treino
NORD_THREADS=2 .venv/bin/python scripts/train_curriculum.py --stage reasoning --steps 100

# 4. Exame final: não treina; mostra os três resultados
NORD_THREADS=2 .venv/bin/python scripts/train_curriculum.py --stage final
```

Cada etapa pode ser repetida para continuar a treinar. O checkpoint está isolado em `~/.nord-brain/curriculum/curriculum-four.pt`, para não confundir com o protótipo anterior `curriculum.pt` ou com V1/V2/V3.

**Interpretação correta dos resultados:** o vocabulário apresenta as 30 palavras com menor perda de previsão entre 250 candidatas frequentes do corpus. Não demonstra que o modelo compreenda os seus significados nem é uma contagem de palavras dominadas. A etapa de frases mostra geração livre, que poderá ainda ser incoerente. O raciocínio usa contas simples; não demonstra raciocínio geral. A fase final apenas executa as avaliações, sem alterar pesos.

**Limitações atuais:** o tokenizer continua híbrido palavra/byte (não BPE), o treino de palavras não ensina explicitamente semântica, os testes são pequenos, o treino de raciocínio é supervisionado e não existe aprendizagem por recompensa. O código ainda não foi validado na VM do utilizador.
