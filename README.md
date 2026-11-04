# Nord Brain 0.1 — laboratório de aprendizagem neuronal no terminal

**Projeto experimental.** Substitui a antiga plataforma Ollama por um pequeno cérebro neuronal próprio, treinável localmente. Não é um assistente geral nem aprende português fluentemente apenas com uma lista de palavras. O objetivo desta versão é provar que os **pesos mudam**, que os conhecimentos persistem e que conseguimos avaliar a generalização, antes de investigar crescimento autónomo.

## Iniciar (Linux, Windows ou macOS)

Requer Python 3.10+ e NumPy. O projeto não usa Qwen, Ollama, LLM externo nem serviço web obrigatório.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m nord_brain
```

No Linux também podes executar `bash install.sh` e depois `.venv/bin/python -m nord_brain`. Windows: `.venv\Scripts\python -m nord_brain` após instalar NumPy no venv. Por defeito, os ficheiros de aprendizagem ficam em `~/.nord-brain/`; usa `--data ./dados` para escolher a pasta.

## Ensinar pelo terminal

```text
nord> /ajuda
nord> /ensinar olá => olá!
nord> /ensinar quanto é dois mais dois? => quatro
nord> /perguntar olá
nord> /neuronal olá
nord> /conceito água = líquido que bebemos
nord> /praticar 10
nord> /teste quanto é três mais dois? => cinco
nord> /avaliar
nord> /cerebro
nord> /crescer 16
nord> /sair
```

`/perguntar` pode mostrar respostas exatas da memória, **identificadas como tal**. `/neuronal` ignora essa memória e tenta gerar diretamente com a rede, permitindo perceber se o modelo aprendeu a generalizar. A rede usa bytes UTF-8 e não tem respostas programadas, mas na V0.1 o texto gerado pode ser incoerente mesmo após muitos exemplos.

## Estrutura

- `nord_brain/core.py`: rede recorrente byte-level em NumPy (embeddings + tanh RNN + projeção), descida de gradiente real com BPTT e AdaGrad; checkpoint `.npz` atómico; expansão funcionalmente conservadora de neurónios ocultos.
- `nord_brain/runtime.py`: ensina, recupera, pratica, avalia, persiste.
- `nord_brain/storage.py`: SQLite para lições, conceitos, testes separados, métricas e observações externas.
- `nord_brain/vocabulary.py`: sementes de vocabulário português **simbólicas**; não equivalem a linguagem aprendida nos pesos.
- `nord_brain/internet.py`: consulta apenas a API HTTPS da Wikipédia em português, com timeouts, limites e sem seguir redirects.
- `nord_brain/cli.py`: interface no terminal.

## Internet opcional, sem treino automático

Por defeito, está **desligada**. Para permitir pesquisa manual:

```bash
NORD_INTERNET=on python -m nord_brain
```

Com `/pesquisar água`, os títulos e excertos são guardados como **fontes não verificadas** no SQLite. Não são executados como comandos nem atualizam os pesos da rede. A conexão é apenas de saída, restrita a uma origem HTTPS. Uma futura versão pode acrescentar leitura de outros sites, sob controlo de permissões e validação.

## Limites reais

- A rede inicial tem aproximadamente dezenas de milhares de parâmetros e treino online limitado no CPU. **Não** aprende inteligência geral, raciocínio humano nem linguagem adulta a partir de poucos exemplos.
- O treino atual é supervisionado pelo professor; pesquisa e vocabulário são apoios, não substituem aprendizagem neuronal.
- `/crescer` é **manual e experimental**. Adiciona novos pesos mantendo inicialmente a função antiga. Isso não garante melhores respostas e os canais adicionais precisam de treino. Decidir crescer automaticamente deve depender de avaliações robustas ainda por construir.
- Os testes `/teste` ficam separados das lições. É possível que a taxa de sucesso inicial seja 0%. Isso é um resultado válido, não é escondido.
- Mantém cópias de segurança da pasta de dados. A atualização da versão antiga muda o modo de execução: o antigo painel web deixa de ser o ponto de entrada nesta versão.

## Testes

```bash
python -m unittest discover -s tests -v
```

A versão anterior continua acessível no histórico Git; os seus pesos Ollama, histórico e ficheiros não são apagados automaticamente do servidor. **Não execute o antigo `install.sh` em produção sem rever esta migração de aplicação web para CLI.**