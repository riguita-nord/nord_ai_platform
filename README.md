# Nord Brain 0.2 — Transformer próprio e aprendizagem pelo terminal

**Protótipo experimental de investigação.** O núcleo atual já não é uma RNN, nem depende de Qwen, Ollama ou pesos de modelos externos. É um **Transformer decoder-only com pesos inicializados de raiz**, inspirado em componentes arquiteturais utilizados pela família Qwen: **atenção causal com GQA, RoPE, RMSNorm e MLP SwiGLU**. Não tem qualquer conhecimento linguístico treinado inicialmente, pelo que as primeiras gerações podem ser incoerentes.

A versão 0.2 prova funcionalidades técnicas: treino real dos parâmetros, memória persistente, causalidade, crescimento da MLP sem alterar de imediato a função e armazenamento de experiências. **Não demonstra inteligência geral, fluência em português, autonomia de investigação nem crescimento inteligente automático.**

## Instalar e iniciar

Requer Python 3.10+ e PyTorch 2.2+ com suporte a CPU. Em Linux, recomenda-se o PyTorch **CPU-only**, pois as rodas PyPI padrão podem descarregar dependências CUDA desnecessárias. Numa máquina com GPU consulta a instalação PyTorch apropriada ao hardware.

```bash
git clone https://github.com/riguita-nord/nord_ai_platform.git
cd nord_ai_platform
bash install.sh
.venv/bin/python -m nord_brain
```

O `install.sh` instala PyTorch CPU-only em Linux. Para instalação manual ou noutros sistemas:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m nord_brain
```

Windows: ativa `.venv\Scripts\activate` e executa `python -m nord_brain`. As dependências de PyTorch ainda podem ocupar bastante disco e memória: mesmo o preset `tiny` **não é adequado a um ESP32** nem constitui um runtime C++ de produção.

### Perfis

O modelo selecionado na **primeira execução** fica preservado no checkpoint, mesmo que mudes a variável posteriormente:

| Preset | Camadas | Dimensão | Attention / KV | MLP | Contexto | Aproximadamente |
|---|---:|---:|---:|---:|---:|---:|
| `tiny` | 2 | 64 | 2 / 1 | 192 | 128 | 0,12M pesos |
| `small` (padrão) | 4 | 128 | 4 / 2 | 384 | 256 | 0,82M pesos |
| `medium` | 6 | 192 | 6 / 3 | 512 | 384 | 2,49M pesos |

```bash
NORD_BRAIN_PRESET=tiny .venv/bin/python -m nord_brain --data ./meu-cerebro
```

Para mudar de arquitetura de forma segura, usa uma **pasta de dados nova**. Os pesos de diferentes tamanhos não são intercambiáveis automaticamente.

## Interagir e ensinar

```text
nord> /ajuda
nord> /cerebro
nord> /ensinar olá => olá!
nord> /ensinar como te chamas? => Sou a Nord Brain.
nord> /perguntar olá
nord> /neuronal olá
nord> /praticar 10
nord> /teste quanto é 3 mais 2? => cinco
nord> /avaliar
nord> /crescer 16
nord> /guardar
nord> /sair
```

- `/ensinar` modifica **efetivamente** os pesos, com gradient descent (AdamW), usando a sequência de resposta como alvo de treino. Também guarda a experiência em SQLite para revisão.
- `/perguntar` pode devolver a resposta **exata da memória simbólica**: a origem é identificada e não conta como compreensão neuronal.
- `/neuronal` ignora a memória e tenta gerar apenas com os pesos do Transformer. As respostas podem ser inválidas.
- `/teste` regista exemplos **não incluídos no treino** e bloqueia ensinar exatamente essas perguntas. `/avaliar` usa geração determinística para medir respostas exatas, mas a avaliação é limitada e não prova raciocínio geral.
- `/crescer 16` acrescenta 16 canais em cada MLP preservando inicialmente a saída; reinicia o estado do otimizador porque as matrizes mudaram de dimensão. **Não acrescenta conhecimento por si só.**

O tokenizer inicial é **UTF-8 por bytes** (258 IDs, incluindo dois tokens de controlo). Qualquer texto português é representável sem reconstruir vocabulário. Um ficheiro de **493 sementes lexicais** ajuda a futura aprendizagem orientada, mas **não** treina os pesos por si só; a compreensão da língua não nasce do dicionário.

## Componentes da arquitetura

- `nord_brain/core.py` — Transformer, RMSNorm, RoPE, GQA, SwiGLU, treino, geração, expansão e checkpoint com modelo + AdamW.
- `nord_brain/runtime.py` — orquestra o cérebro, replays, memória, avaliações e perfis.
- `nord_brain/storage.py` — SQLite: lições, conceitos, observações, métricas e testes separados.
- `nord_brain/vocabulary.py` — palavras iniciais com categorias, não pré-treino.
- `nord_brain/internet.py` — pesquisa manual limitada a Wikipédia em português com timeout, limites e URL de origem.
- `nord_brain/cli.py` — interface de terminal sem servidor web.
- `tests/test_brain.py` — causalidade, treino, persistência, expansão e separação memória/rede.

## Internet opcional

Desligada por defeito. Liga com:

```bash
NORD_INTERNET=on .venv/bin/python -m nord_brain
```

O comando `/pesquisar água` consulta somente uma origem HTTPS permitida, guarda título, URL e um excerto, marcado como **não verificado**. **Não há treino automático a partir da Internet**, nem execução do conteúdo de páginas. Futura integração de fontes, experiências ou navegadores deve manter isolamento, validação e autorizações.

## Limitações e migração

- A aprendizagem é supervisionada e incremental: precisa de **muitos exemplos e avaliações** para desenvolver linguagem útil. A taxa de perda diminuir não implica compreensão.
- O crescimento é manual e limitado às MLPs; não existe ainda escolha automática de quando crescer, poda sináptica ou um sistema provado de aprendizagem geral ao longo da vida.
- O checkpoint V2 é `~/.nord-brain/brain-v2.pt`, preservando intacto o antigo `brain.npz` RNN V0.1. **Não existe conversão entre pesos RNN e Transformer.** O vocabulário, a memória SQLite e as lições da V0.1 são mantidos, mas o novo modelo neuronal começa do zero.
- Esta versão **não mantém o painel web/Ollama anterior nem altera automaticamente serviços systemd**. O serviço da antiga plataforma pode ainda estar instalado no servidor. Faz backup e revê o serviço antes da migração para CLI.
- O ambiente de teste com PyTorch não equivale a um runtime embebido final em C++. Numa VM com 2 GB RAM e 1 vCPU, o modo `tiny` é o primeiro a testar, mas o consumo real depende da distribuição e da instalação do PyTorch.

## Testes

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Consulta o histórico Git para recuperar a versão anterior da plataforma, se necessário.
