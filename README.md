# Nord Brain V2.0 — Transformer próprio, sem segundo modelo

**O cérebro que responde é o Nord Transformer treinado pela própria Nord Brain.**
Por defeito, a aplicação **não usa Qwen, llama.cpp, APIs externas de IA nem
downloads de modelos**. A janela de memória da conversa, a lógica verificada e
os mundos simulados continuam a integrar o mesmo cérebro.

## Começar

```bash
git pull origin main
./nord
```

`/diagnostico-linguagem` mostra o estado do gerador nativo.
`/gerar-neural conta-me uma história` mostra texto bruto criado diretamente
pelos pesos, mesmo que ainda não seja português correto.

## Ensinar linguagem real aos pesos

O treino anterior usava sobretudo exemplos curtos de pergunta/resposta. A
versão V2.0 adiciona **modelação causal de texto corrido**: o Transformer tenta
prever os bytes seguintes de narrativas, diálogos e parágrafos previamente
**aprovados por ti**. Cada sessão treina no máximo 32 pequenas janelas em CPU.

1. Escreve ou escolhe um ficheiro UTF-8 `.txt` ou `.md` (entre 180 B e 100 KB).
2. Na Nord Brain, escreve:

```text
/treinar-texto ~/portugues/historias.txt 6
/gerar-neural conta-me uma história
/avaliar-portugues
```

O sistema guarda os pesos e o otimizador, regista o hash do documento e
**reverte a sessão** se os testes de lições do professor ou de exemplos
reservados sofrerem uma regressão acima dos limites. Não treina
automaticamente sobre informação da Internet nem sobre as próprias histórias
geradas. Os dados de treino não substituem os factos que lhe ensinaste.

Para depuração, `NORD_NATIVE_MIN_STEPS=80` define o mínimo de atualizações
antes de experimentar geração neuronal num diálogo normal; o comando
`/gerar-neural` continua a mostrar a saída bruta.
`NORD_NATIVE_GENERATION=off` desativa a geração nativa sem desligar
a memória nem os verificadores. A integração histórica com um modelo
pré-treinado só é acessível se o operador selecionar **explicitamente**
`NORD_LANGUAGE_BACKEND=external`; não faz parte do percurso normal.

## Limitação real

O modelo nativo tem pesos próprios, mas um Transformer muito pequeno,
inicializado do zero, **não fica fluente só por receber um novo motor de
contexto**. A versão V2.0 melhora as vias de treino e permite experimentar
a geração autónoma sem outro modelo. Quando a geração não atinge critérios
mínimos de qualidade, a Nord Brain mantém os módulos verificados ou a
composição procedural já existentes. Esses textos não são apresentados como
prova de criatividade neuronal. Para aproximar a escrita de um modelo
avançado, será necessário um corpus português muito maior, treino prolongado
e avaliação independente em conversas e histórias nunca vistas.

---

# Nord Brain V1.9 — instruções incompletas e continuidade real

Quando um pedido curto acrescenta algo ao que acabou de ser dito — por exemplo,
**«com capítulos e etc.»**, **«sem spoilers»**, **«em tópicos»** ou
**«com mais diálogos»** — a Nord Brain tenta interpretá-lo como alteração
ao assunto atual, e **não como uma resposta para treinar**.

Exemplo:

```text
tu> conta-me uma historia
nord> [conto original]
tu> faz mais extenso
nord> [mesma história desenvolvida]
tu> com capítulos e etc
nord> [Capítulo 1, 2, 3 da história anterior; livro guardado]
tu> continua a história
nord> [Capítulo 4 do mesmo livro]
tu> plano do livro
nord> [plano e progresso persistente]
```

A conversão para capítulos preserva os parágrafos existentes e guarda as
personagens, o cenário e a descoberta como base para futuros capítulos.
Fica em `~/.nord-brain/stories.sqlite3`, sem substituir lições nem pesos.
O modelo local de linguagem, quando instalado, pode reescrever conteúdo e
entender alterações de estilo mais livres; sem ele, as reformulações são
limitadas aos tipos de transformação que o sistema suporta. **A memória
de contexto não torna o pequeno Transformer fluente por si só.**

O histórico de conversas mantém-se separado da aprendizagem. A versão V1.9
inclui testes de regressão para pedidos elípticos e continuidade entre
mensagens. Para verificar: `python -m unittest discover -s tests -p
'test_context_ellipses_v19.py' -v`.

---

# Nord Brain V1.8 — contexto conversacional por assunto

A memória de conversa é partilhada por **todos os módulos**. A Nord Brain mantém até **480 interações locais** em `~/.nord-brain/dialogue.sqlite3`, recuperando as **oito interações recentes** e até **três interações antigas relevantes** para cada nova mensagem ao chamar o modelo local. A recuperação é lexical e leve para CPU, não equivale a compreensão semântica perfeita.

```text
tu> conta-me uma história
nord> [história sobre Nuno e uma máquina misteriosa]
tu> faz mais extenso
nord> [a mesma história, desenvolvida]
tu> quem era ele?
nord> [responde usando a personagem da história]
tu> consegues fazer maior
nord> [recupera a história completa, não a resposta curta anterior]
```

Também funciona para projetos, problemas e investigações: quando voltas a mencionar um assunto antigo, o histórico relevante pode ser recuperado mesmo depois de outras conversas e de reiniciar a Nord Brain. O modelo pré-treinado local continua a ser necessário para interpretação livre, reformulações e respostas naturais; o pequeno Transformer experimental não herda essa capacidade só por receber memória.

Pedidos de continuação e reformulação nunca são exemplos de treino automático, mesmo quando existia uma pergunta sem resposta pendente. Usa `/contexto` para o último assunto e `/limpar-contexto` para apagar apenas esta memória. As lições ensinadas, livros e checkpoints não são alterados pela limpeza.

## Histórico V1.7 — contexto contínuo para toda a conversa

A V1.7 mantinha **um contexto local e limitado das 24 últimas interações** (`~/.nord-brain/dialogue.sqlite3`), separado de `memory.sqlite3` e dos pesos neuronais. Não é um novo modelo pré-treinado: é memória operacional para interpretar referências a mensagens anteriores.

```text
tu> conta-me uma história
nord> [história original]
tu> faz uma mais longa
nord> [reescreve a MESMA história, mantendo personagens, descoberta e conflito]
tu> continua a história
nord> [continua os acontecimentos anteriores]
tu> muda o final
nord> [novo final para a história que está a ser discutida]
tu> qual era o nome da personagem?
nord> [responde usando a história anterior]
```

Também funciona com livros criados por `cria um livro sobre ...`: depois de escrever um capítulo, `faz uma mais longa` atualiza o **mesmo capítulo guardado**, sem criar um livro novo. `continua a história` escreve o capítulo seguinte. O contexto continua disponível depois de reiniciar o terminal, enquanto usares a mesma pasta de dados.

- **Com o Qwen local ativo:** reescreve e prolonga a história com um pedido contextual ao modelo, limitado para CPU.
- **Sem modelo pré-treinado:** cria uma expansão procedural que preserva a narrativa anterior; a criatividade e a naturalidade continuam limitadas. Não a apresenta como fluência neuronal.
- **Outros assuntos:** `explica melhor`, `fala mais sobre isso` e semelhantes utilizam as respostas recentes como contexto; sem modelo local, a Nord Brain identifica o tema anterior e assinala a limitação.
- **Privacidade e controlo:** `/contexto` mostra o último assunto, `/limpar-contexto` apaga o histórico de conversa e o histórico transitório do modelo linguístico. **Não apaga** livros, lições, checkpoints ou a memória do professor.
- A conversa **nunca é convertida automaticamente em factos ou exemplos de treino**. Pedidos como `faz uma mais longa` não alteram os pesos neuronais nem entram nas lições.

Atualiza com `git pull origin main` e inicia com `./nord`. Os testes incluem a reprodução da conversa em que uma história curta é seguida por `faz uma mais longa`, além de testes de persistência, isolamento da memória e reescrita de capítulos.

---


## Conversa geral contínua (V1.7)

- **Todos os módulos contam**: depois de uma conta, pesquisa, simulação, livro ou conversa casual, o texto da interação fica em `dialogue.sqlite3`, separado de `memory.sqlite3` e de qualquer treino neuronal.
- **Mesmo depois de reiniciar**, na V1.7, o modelo local recebia até **cinco interações anteriores** como mensagens `user`/`assistant` na API de conversação, com texto abreviado para poupar RAM/contexto. A versão V1.8 conserva até 480 interações e seleciona as relevantes.
- **Referências e continuidade**: «o que achas disso?», «e ele?», «quem é ele?», «explica melhor», «resume isso», «faz uma mais longa» e «continua» utilizam o assunto anterior em vez de ensinar falsas respostas.
- **Assunto ativo**: perguntas relacionadas herdam o assunto inicial; uma pergunta independente passa a ser o novo assunto. O histórico do Qwen não é uma memória paralela e volátil: usa o mesmo registo de diálogo da Nord Brain.
- **Sem Qwen pré-treinado**: a Nord Brain ainda identifica o assunto anterior e consegue fazer certas revisões procedurais de histórias, mas **não** consegue garantir explicações e conversas livres. Um histórico não dá fluência a um Transformer treinado do zero.
- **Privacidade**: o modelo linguístico só pode ser servido no `localhost`. O histórico fica no disco local. `/contexto` mostra o assunto atual e `/limpar-contexto` apaga só o histórico, mantendo lições e livros.

Teste manual:

```text
tu> 4+4
nord> O resultado é 8.
tu> explica melhor
tu> o que achas disso?
tu> conta-me uma historia
tu> faz uma mais longa
tu> quem era ela?
```

> A conversa não tem memória infinita. O contexto completo dos últimos cinco turnos pode ser resumido por limites de tamanho; conhecimentos importantes devem ficar guardados separadamente e com proveniência.

# Nord Brain V1.6 — romances e histórias longas por capítulos

A Nord Brain agora dispõe de um **estúdio de histórias persistente**. Ao contrário do pedido de conto curto, um livro é desenvolvido em capítulos, com plano de enredo, protagonista, cenário, conflito e resumos do que aconteceu. Os capítulos continuam disponíveis depois de reiniciares o programa. A ficção **não é gravada como conhecimento factual nem usada automaticamente no treino neuronal**.

## Conversa normal

```text
tu> Cria um livro sobre uma máquina que descobre a própria cidade.
nord> Comecei o livro ... (10 capítulos)
      Capítulo 1 — A primeira pista
      [texto do capítulo]

tu> Continua a história
nord> Capítulo 2 — A porta entreaberta
      [novo capítulo, continuando o anterior]

tu> Plano do livro
nord> ... 2/10 capítulos, plano completo ...

tu> Mostra capítulo 1
nord> [capítulo guardado]

tu> Exporta o livro
nord> Manuscrito exportado em Markdown: .../books/livro-001.md
```

Também podes escrever `Conta-me uma história longa com 12 capítulos sobre robôs`, `Cria um romance de 8 capítulos sobre uma aventura` ou `Continua mais 2 capítulos`.

## Comandos /livro

```text
/livro novo O mistério de uma estação abandonada
/livro novo 12 Um robô que aprende a pintar
/livro continuar 2
/livro estado
/livro listar
/livro selecionar 1
/livro ler 3
/livro exportar
```

- **Planeamento:** de 3 a 24 capítulos por livro; 10 por defeito. Cada pedido escreve o primeiro ou até três capítulos adicionais, limitando o trabalho em CPU.
- **Continuidade:** `stories.sqlite3` guarda vários projetos de livros, capítulos e resumos compactos para o modelo continuar sem reenviar um romance inteiro a cada pedido.
- **Texto:** com `llama.cpp`/Qwen local ativo, cada capítulo é gerado pelo modelo com contexto narrativo; sem o modelo, há um **fallback procedural**, de qualidade e variedade muito mais limitadas, mas que não devolve "não sei contar histórias".
- **Tamanho:** `NORD_STORY_CHAPTER_TOKENS=650` por defeito (200–1050) controla a geração por capítulo. Valores maiores consomem mais CPU, RAM/contexto e demoram mais. O modelo pode escrever menos do que o solicitado.
- **Exportação:** o manuscrito Markdown é guardado em `~/.nord-brain/books/livro-NNN.md` (ou na pasta indicada por `NORD_DATA`). O original continua na base SQLite. Não existe envio automático de ficheiros para a Internet.
- **Limites:** o planeamento mantém a estrutura, mas um Qwen 0.6B pode repetir ideias ou contradizer detalhes em livros muito longos. A continuação é um novo pedido de geração; não há escrita em background. Para obter melhor prosa, inicia o servidor linguístico com `bash scripts/start-local-language.sh` e confirma em `/diagnostico-linguagem`.

## Atualização

```bash
cd ~/nord_ai_platform
git pull origin main
./nord
```

Testes locais: `python -m unittest discover -s tests -v` (com as dependências instaladas).

---

# Nord Brain v1.5 — histórias, conversa em português e diagnóstico de pesquisa

Esta atualização corrige o bloqueio de pedidos criativos e a falta de variedade nas saudações e acrescenta uma segunda pesquisa sem chave. **O Transformer original da Nord Brain continua experimental** e não se transforma num modelo fluente só porque a perda diminuiu. Para conversa e escrita realmente naturais, a V1.5 pode usar **um modelo Qwen3-0.6B quantizado localmente em CPU**, mantendo o cérebro próprio, a memória e os seus pesos separados.

## Como testar já, mesmo sem Qwen

```bash
cd ~/nord_ai_platform
git pull origin main
./nord
```

Na conversa: `ola`, `ola`, `ola` (saudações variadas), `conta-me uma história`, `inventa um conto sobre robôs`, `dá-me ideias para um livro`, `2+2` e `quem é Elon Musk?`. Sem modelo pré-treinado, as histórias e ideias são uma **composição procedural simples**, nunca inteligência criativa equivalente a Qwen. O pedido deixa de ser tratado como resposta de treino. A aritmética continua verificada e o ensino humano tem prioridade na memória.

## Português e histórias com um modelo pré-treinado na CPU

Numa segunda janela de terminal, **por tua iniciativa**, executa:

```bash
cd ~/nord_ai_platform
bash scripts/start-local-language.sh
```

O script descarrega `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf` (~397 MB, Apache-2.0), verifica o SHA-256, encontra ou compila `llama-server` (precisas de `git`, `cmake`, `c++` e `curl`) e abre **apenas em `127.0.0.1:8081`** com CPU (`-ngl 0`). A instalação e o download não acontecem silenciosamente durante `./nord`. O compilador pode precisar de pacotes como `build-essential` e `cmake`; num Pi com 2 GB de RAM o desempenho e a simultaneidade com PyTorch podem ser limitados. Idealmente usa um Pi com 8 GB ou mais.

Na primeira janela:

```bash
cd ~/nord_ai_platform
NORD_SHOW_ORIGIN=on ./nord
```

O cliente usa automaticamente um serviço **local** compatível com `POST /v1/chat/completions` se existir. Conversas e histórias passam pelo Qwen local quando disponível; respostas baseadas em conhecimento ensinado e cálculos verificados mantêm prioridade. O histórico curto enviado ao Qwen serve apenas para continuidade da conversa, **não atualiza os pesos**. Os pesos do Transformer Nord continuam a ser treinados e avaliados separadamente.

Para verificar se o Qwen está ligado escreve `/diagnostico-linguagem`. `NORD_LLM=off ./nord` força o modo experimental sem o modelo pré-treinado. `NORD_LLM_URL` só aceita endpoints HTTP em loopback local e nunca envia automaticamente conversa para um servidor remoto.

## Pesquisa com causas de falha visíveis

A pesquisa agora tenta a fonte primária selecionada, a Brave Search API se a chave `NORD_BRAVE_API_KEY` estiver configurada, a página HTML do DuckDuckGo e, por fim, o RSS público do Bing. As duas últimas vias são **não oficiais e podem ser bloqueadas**; não é possível prometer resultados sem uma API contratada.

```text
/diagnostico-pesquisa
/diagnostico-pesquisa Elon Musk
```

O diagnóstico indica quantos resultados úteis cada fornecedor devolveu ou que tipo de erro ocorreu. Se quiseres usar a API oficial, configura `NORD_BRAVE_API_KEY` no teu ambiente. Todas as fontes externas são guardadas como evidências provisórias: **nem mesmo documentação oficial entra automaticamente nos pesos**. Só uma confirmação tua permite usar o excerto para treino. Um conhecimento explicitamente ensinado por ti prevalece sobre um excerto externo mais recente, e uma contradição é assinalada em vez de ser substituída silenciosamente.

**Limitações honestas:** não se demonstrou fluência do Transformer criado do zero; a criatividade fluente do Qwen provém dos pesos pré-treinados desse modelo separado. Este lançamento não implementa treino LoRA do Qwen em Raspberry Pi, navegação web garantida nem um navegador completo que faça pesquisa aberta em segundo plano. Pesquisa factual falhada não é substituída por palpites do modelo.

---

# Nord Brain V1.2 — pesquisa pública com fontes e aprendizagem textual controlada

A Nord Brain mantém o Transformer experimental próprio, a matemática verificável e o laboratório de hipóteses da V1.1. **O modelo de texto continua sem pré-treino equivalente ao Qwen**, por isso é perigoso tratar palavras aleatórias como respostas. A V1.2 passa a distinguir perguntas públicas de texto ensinável e a apresentar fontes quando não tem conhecimento fundamentado.

## Resolver as perguntas do terminal

```text
tu> e 3+5
nord> O resultado é 8. 3 + 5 = 8

tu> quem é Elon Musk?
nord> Encontrei estas informações sobre Elon Musk (ainda não confirmadas):
      [1] [Título e excerto devolvidos pelo motor de pesquisa]
          Fonte: https://... (endereço real)

 tu> confirma fonte 1
nord> Confirmação registada. Atualizei os pesos com a informação que validaste...

tu> experimenta linguagem 2
nord> Laboratório de experiências textuais com as definições aprovadas.
```

Não há um resumo fixo sobre Elon Musk no código. A resposta depende de **resultados reais** e mostra os links. Caso não encontre fontes, diz que não sabe em vez de inventar `3` ou `t`. Confirmar um resultado atualiza os pesos PyTorch (passo de gradiente), mas **isso não garante que a rede consiga generalizar noutras frases**. O laboratório de linguagem pode criar perguntas novas a partir das definições validadas e comparar previsão/resultado com questões separadas para avaliação.

## Pesquisa pública além de quatro sites

- Para documentação de Python, Raspberry Pi, NASA e kernel Linux, mantém fontes primárias fixas previamente autorizadas; esses resumos curtos podem continuar a treinar automaticamente quando `NORD_SOURCE_TRAIN=on` (comportamento anterior).
- Para assuntos gerais, pessoas e outros temas: usa a API Brave Search com chave opcional `NORD_BRAVE_API_KEY`, se configurada. Sem chave tenta os resultados HTML sem JavaScript do DuckDuckGo. Esta segunda via **não é uma API suportada** e pode bloquear pedidos ou deixar de funcionar, por isso não há garantia de disponibilidade.
- Os resultados públicos são considerados **excertos não verificados**; não são treino automático. O utilizador escolhe `confirma fonte N` antes de os tratar como exemplos ensinados. Para corrigir, pode explicar diretamente um facto ou rejeitar uma hipótese.
- É mostrada a URL de cada fonte. Wikipedia/Wikimedia permanecem excluídos segundo a preferência do projeto. Páginas externas não são executadas nem abertas para leitura livre: só se usam os excertos fornecidos pelo motor de pesquisa. O motor de pesquisa recebe o tema público, não o histórico da conversa.
- Pesquisa requer acesso HTTPS de saída na VM. A existência de uma fonte não significa que a afirmação esteja correta, nem que a Nord Brain tenha aprendido a linguagem naturalmente.

```bash
# atualiza e arranca
cd ~/nord_ai_platform && git pull origin main && bash install.sh

# opcional: definir chave do serviço de pesquisa na shell (não a guardes no git)
export NORD_BRAVE_API_KEY='A_TUA_CHAVE'
./nord

# modo sem Internet
NORD_INTERNET=off ./nord
```

Para verificar o próprio cérebro sem a ajuda do verificador matemático, consulta os resultados neuronais de `/avaliar`, `/cerebro` e `NORD_SHOW_ORIGIN=on ./nord`. A aprendizagem e as simulações em CPU funcionam enquanto a aplicação está aberta.

---

# Nord Brain V1.1 — teorias de causa e efeito e controlo do professor

Agora, depois de uma demonstração verificada como `2+2 é 4`, a Nord Brain gera **hipóteses contrafactuais**: mudar o primeiro ou segundo valor em +1/-1, transferir 1 entre eles, trocar a operação, duplicar ambos e experimentar novos valores. Por exemplo, `3+2=5`, `1+2=3`, `3+1=4`, `3+3=6` e `4+4=8`. Cada hipótese tem a explicação da transformação, é prevista pelo Transformer antes de se conhecer o resultado, verificada independentemente por matemática determinística e só depois pode atualizar os pesos. O verificador não é o Transformer; uma resposta correta do verificador **não prova generalização neuronal**.

## Dar controlo à pessoa que ensina

Escreve naturalmente no terminal:

```text
tu> quero aprovar as hipóteses
nord> Modo de revisão ligado. As hipóteses matemáticas aguardam aprovação.

tu> 2+2 é 4
nord> ... preparei hipóteses para verificar e aprender ...

tu> mostra teorias
nord> Quanto é 3 + 2? = 5 | review | e se acrescentar 1 ao primeiro número?

tu> aprova hipótese 3+2
nord> Aprovada. Será matematicamente verificada antes do treino.

tu> rejeita hipótese 4+4
nord> Bloqueei essa hipótese para novas revisões e atualizações de pesos.

tu> pausa aprendizagem
nord> Parei o treino autónomo e as simulações automáticas.

tu> retoma aprendizagem
nord> Reativei as experiências e o treino autónomo.
```

O modo de revisão de hipóteses matemáticas é **opcional**; por defeito, as hipóteses aritméticas verificadas treinam autonomamente. A pausa aplica-se a todas as atualizações autónomas, incluindo o simulador de movimento e a investigação. As instruções explícitas do professor continuam a poder ensinar mesmo com a autonomia pausada. O modo de revisão não bloqueia automaticamente o currículo autónomo de outras áreas; usa `pausa aprendizagem` para parar todo esse treino.

Se disseres `3+2 é 8`, a conta é recusada pelo verificador; `3+2 é 5` é aceite. **Rejeitar uma hipótese depois de já ter sido treinada bloqueia experiências futuras, mas não desfaz os pesos anteriormente modificados.** A fila, as decisões do professor e os bloqueios sobrevivem ao reinício da aplicação.

Existe ainda o laboratório comum da versão anterior: `experimenta tudo` (matemática, padrões, lógica, linguagem e mundo virtual) ou `experimenta lógica 2`. Na VM pequena, reduz experiências por ciclo usando as variáveis de ambiente indicadas abaixo.

# Nord Brain V0.9 — hipóteses matemáticas e investigação com fontes primárias

> **Experimental:** a Nord Brain tem dois subsistemas de aprendizagem: um Transformer linguístico pequeno com pesos treináveis e uma rede pequena que prevê consequências no mundo virtual. Nenhum deles tem o conhecimento inicial de um Qwen pré-treinado. Os verificadores fazem contas corretas, mas isso **não prova que o Transformer aprendeu matemática**. O desempenho neuronal é medido apenas com perguntas que não entraram no treino.

## Como começar a ensinar

```text
tu> 2+2 é 4
nord> Correto: ... Registei o exemplo verificado. Preparei novas hipóteses para experimentar e aprender nos próximos ciclos.

tu> 3+1
nord> O resultado é 4. [origem: cálculo verificável; tentativa neuronal correta/incorreta]

tu> o que sabes
nord> ... Hipóteses matemáticas: X por testar, Y verificadas e treinadas ...
```

- A afirmação do professor é verificada primeiro. Se disseres `2+2 é 5`, o cérebro rejeita-a.
- Uma conta correta desencadeia um laboratório de **hipóteses matemáticas**: diferentes somas que dão 4 (como `3+1`, `1+3`, `4+0`), subtrações, multiplicações e exemplos com resultados **diferentes** (3 e 5). O objetivo é aprender regras, não repetir sempre 4.
- Cada hipótese é guardada numa fila SQLite, **prevista primeiro pelo Transformer**, depois avaliada pelo verificador matemático. Só as experiências verificadas atualizam os pesos. A fila sobrevive aos reinícios.
- O ciclo autónomo continua a gerar contas com novos números, separados entre treino e testes. Um resultado do verificador não conta como um acerto neuronal; o avanço para níveis mais difíceis depende de acertos reais da rede em exemplos não vistos.
- As experiências matemáticas continuam apenas enquanto o processo estiver aberto. Os conhecimentos do utilizador, os checkpoints e o histórico de hipóteses são persistentes.

## Pesquisa autónoma na Internet — sem Wikipédia

A Internet está **ativada por defeito** para investigação limitada a documentação primária selecionada. Só faz pedidos de rede quando pedes uma pesquisa ou existe uma dúvida pendente sobre um destes assuntos:

| Assunto | Fonte | Tipo |
|---|---|---|
| Python | `docs.python.org` | Documentação oficial da linguagem |
| Raspberry Pi | `raspberrypi.com` | Documentação do fabricante |
| Sistema solar e planetas | `science.nasa.gov` | NASA Science |
| Linux/kernel | `docs.kernel.org` | Documentação dos autores do kernel |

**Segurança e honestidade:** o código não aceita URLs arbitrárias, redirecionamentos nem websites desconhecidos; não envia a frase completa do utilizador a motores de busca externos, não executa JavaScript e limita a leitura a 120 KB por fonte, com timeout. Guarda URL e proveniência. Um pequeno excerto oficial pode ser usado **uma única vez** para atualizar os pesos, mas não é tratado como facto confirmado por fontes independentes; a rede pode continuar a gerar erros.

```bash
# Normal: Internet oficial e treino com pequenos excertos de fontes ativados
./nord

# Desligar totalmente a Internet
NORD_INTERNET=off ./nord

# Permitir pesquisa mas impedir que o texto da fonte seja treinado
NORD_SOURCE_TRAIN=off ./nord

# Mostrar que resposta veio de cálculo, memória, dedução ou aprendizagem neuronal
NORD_SHOW_ORIGIN=on ./nord
```

Não há pesquisa aberta no mundo inteiro, validação científica automática nem aprendizagem de todas as páginas. Para adicionar outros sites fiáveis é necessária uma revisão explícita das fontes no código. O treino sobre fontes externas é uma **experiência de aprendizagem por dados**, não uma garantia de veracidade ou fluência. A execução na Raspberry Pi continua limitada pelo CPU/RAM.

## Simulador de ações

Além de contas, a V0.9 inclui o `world.py` que simula uma máquina em grelha. A rede observa sensores, prevê consequências de quatro ações, experimenta uma, recebe feedback e ajusta pesos. A simulação é local e não controla motores físicos. Podes dizer `simula 3 cenários` na conversa.

## Atualização

```bash
cd ~/nord_ai_platform
git pull origin main
bash install.sh
```

Para executar a suite local: `.venv/bin/python -m unittest discover -s tests -v`. Não é necessário apagar `.venv`, os dados SQLite nem os checkpoints.

---

# Nord Brain V0.5 — aprendizagem autónoma com testes de generalização

**Objetivo:** deixar de depender de corrigir exemplos individuais. O Transformer tenta aprender uma operação a partir de múltiplas experiências diferentes, não de uma tabela de perguntas/respostas. O treino é real (gradientes e pesos PyTorch), mas um modelo com pesos iniciados do zero pode continuar a errar mesmo depois de muitas experiências.

## Como funciona a aprendizagem autónoma

- Gera contas novas de forma procedural a partir de números e operações, com respostas calculadas por um verificador independente. Não existe uma lista fixa de respostas para `2+2` ou `4+4`.
- Reserva de forma determinística parte das combinações **fora do treino**. A avaliação de competência utiliza **apenas a geração do Transformer** nesses exemplos, nunca o verificador matemático.
- Treina em grupos de quatro exercícios por ciclo, de 30 em 30 segundos por defeito, para continuar viável numa VM de 1 vCPU. O ritmo é configurável em `NORD_MATH_EXAMPLES_PER_TICK` (1–12) e `NORD_AUTO_INTERVAL` (5–3600).
- Depois de obter pelo menos 75% de respostas exatas em testes novos de um nível, avança para um currículo mais difícil: somas pequenas, operações até 20, números até 100 e multiplicações. Sem resultados, **não anuncia que aprendeu** nem desbloqueia o nível por tempo decorrido.
- Ao receber `4+4`, `quanto é 4+4?` ou outras formas de pedido, o núcleo tenta calcular **pelos pesos** e confronta o seu resultado com um cálculo exato. Se falhar, responde com o cálculo verificado. Mostra `NORD_SHOW_ORIGIN=on` para distingui-los; o sucesso do verificador não é contabilizado como sucesso neuronal.
- Mantém o sistema anterior de memória, deduções e crescimento neuronal experimental. Nenhuma pesquisa web não verificada entra automaticamente no treino.

**Limites reais:** o verificador sabe fazer cálculos por programação; isso é uma ferramenta, não raciocínio emergente do modelo. Mesmo com um currículo variável e avaliação em exemplos inéditos, o Transformer muito pequeno pode não generalizar com facilidade. É necessário treino extenso, dados de qualidade, memória de longa duração e testes rigorosos para aproximar a qualidade de um LLM. A aprendizagem autónoma decorre apenas enquanto o processo Nord Brain está em execução.

## Atualização

```bash
cd ~/nord_ai_platform
git pull origin main
bash install.sh
```

Para acompanhar a diferença entre cálculos verificados e capacidade do Transformer, executa `NORD_SHOW_ORIGIN=on ./nord` e pergunta `o que sabes`.

---

# Nord Brain V0.4 — raciocínio verificável, aprendizagem autónoma e conversa pelo terminal

**Projeto experimental, sem pesos pré-treinados.** O cérebro é um Transformer próprio em PyTorch (atenção causal/GQA, RoPE, RMSNorm e SwiGLU) e **não usa o Qwen nem Ollama**. A arquitetura está preparada para aprender pesos, guardar experiências e experimentar expansão neuronal. **Não é equivalente a um LLM treinado; não compreende português fluentemente e não existe garantia de inteligência geral.**


## Novo na V0.4 — Raciocínio com provas e treino derivado

**O objetivo desta versão é melhorar o raciocínio de forma verificável, sem fingir que o pequeno Transformer já sabe raciocinar como uma pessoa.** O novo `reasoning.py` combina duas vias: um motor simbólico limitado que consegue demonstrar algumas conclusões, e a aprendizagem neuronal existente que recebe também exemplos derivados de premissas ensinadas por ti. O resultado de uma dedução simbólica **não conta como prova de raciocínio da rede neuronal**.

### Ensinar relações e perguntar naturalmente

```text
tu> Luna é um gato.
nord> Aprendi ...
tu> Todo gato é um mamífero.
nord> Aprendi ...
tu> Todos os mamíferos são animais.
nord> Aprendi ...
tu> Luna é um animal?
nord> Sim. Dedução: Luna é um gato → Todo gato é um mamífero → Todos os mamíferos são animais.

tu> Se Rui é um atleta e todo atleta é uma pessoa, Rui é uma pessoa?
nord> Assumindo as duas hipóteses: Sim. ...

tu> Quanto é 2 + 3 × 4?
nord> O resultado é 14. Passos: 3 × 4 = 12; 2 + 12 = 14
```

Também aceita relações de atributos, como `Todos os gatos têm pelo.` e a pergunta `Luna tem pelo?`, além de um modelo muito restrito de problemas de quantidades. Só deduz a partir de **relações afirmativas ensinadas pelo professor**; uma resposta inconclusiva significa falta de premissas, não uma negação. Hipóteses de uma pergunta iniciada por `Se ...` são temporárias e **não alteram a memória persistente**.

### Aprendizagem neuronal associada

- As novas afirmações ensinadas continuam a passar pelo treino real dos pesos do Transformer (AdamW), como na V0.3. As relações também ficam guardadas em SQLite para dedução verificável.
- O ciclo autónomo alterna exercícios verificáveis de soma e de **composição aritmética**; mantém exemplos de treino separados de exemplos reservados à avaliação.
- Quando existem relações suficientes do professor, o agente cria alguns exemplos de conclusões com prova e utiliza-os para treinar os pesos, com limites de contexto, tempo e recursos. **Estes dados são consequências lógicas das afirmações do utilizador, não factos científicos confirmados.**
- A aprendizagem neuronal pode continuar a falhar em perguntas novas. Para responder com confiança a questões cobertas pelas regras, a interface utiliza o motor verificável e identifica a origem através de `NORD_SHOW_ORIGIN=on`.

**Limites:** a linguagem compreendida pelo verificador é restrita (`é`, `tem`, `todos os ... são/têm`, expressões aritméticas e alguns problemas simples). Não existe compreensão geral, raciocínio humano nem verificação automática da verdade das afirmações do professor. A expansão de neurónios da V0.3 continua a ser experimental e não depende de deduções sem testes independentes. Excertos curtos de documentação primária podem entrar no treino com origem identificada, sem serem tratados como factos confirmados.

### Testar só o raciocínio

```bash
.venv/bin/python -m unittest tests.test_reasoning -v
```

## Como iniciar

```bash
git clone https://github.com/riguita-nord/nord_ai_platform.git
cd nord_ai_platform
bash install.sh
.venv/bin/python -m nord_brain
```

Para VM de 2 GB e 1 vCPU, testa o modelo mais pequeno numa pasta de dados própria:

```bash
NORD_BRAIN_PRESET=tiny .venv/bin/python -m nord_brain --data ./dados-tiny
```

O Linux instala PyTorch CPU-only. Não suporta ESP32; mesmo com poucos parâmetros, o runtime PyTorch consome recursos adicionais. Um checkpoint `brain-v2.pt` existente continua a ser carregado automaticamente, preservando os pesos da V0.2. Há cópias de segurança antes de experiências de crescimento. Os checkpoints são locais e devem ser considerados ficheiros privados.

## Ensinar como numa conversa

O ensino já não exige `/ensinar`. Alguns exemplos reconhecidos pelo **pequeno interpretador de português**, que é auxiliar ao modelo (não se deve confundir com raciocínio neuronal):

```text
tu> Olá
nord> Olá! Estou a aprender. Podes ensinar-me algo ou fazer uma pergunta.

tu> Chamo-me Ricardo.
nord> Prazer! Aprendi que te chamas Ricardo.

tu> O Python é uma linguagem de programação.
nord> Entendido. Aprendi a descrição de Python.

tu> O que é Python?
nord> Python é uma linguagem de programação.

tu> Quanto é 2 + 2?
nord> Ainda não sei responder. Podes explicar-me?

tu> É 4.
nord> Entendido. Guardei o exemplo e atualizei os pesos neuronais.

tu> O que aprendeste?
nord> Já fiz ... passos de treino ...
```

Também podes dizer `Quando eu perguntar X, responde Y.` e `Não, a resposta correta é Y.` depois de uma pergunta. Estes padrões **atualizam os pesos com PyTorch**, guardam o exemplo e permitem revisões futuras. A resposta a uma pergunta já ensinada pode vir diretamente da memória SQLite. **Isso não demonstra que o Transformer compreendeu a resposta.** Para ver a origem, utiliza `NORD_SHOW_ORIGIN=on`.

O interpretador não entende qualquer frase livre: só reconhece padrões simples de ensino, correção e pergunta. Se encontra uma pergunta que não sabe, pede explicação em vez de fingir certeza. A geração neuronal sem pré-treino pode ser incoerente.

## Aprendizagem realmente automática

Um agente local executa um ciclo de aprendizagem por defeito a cada 30 segundos, enquanto o processo está aberto:

1. Escolhe um exercício simples de aritmética com resposta verificável. Um conjunto de casos separados **não é usado para ensinar**.
2. Efetua um pequeno passo de treino dos pesos, conserva o checkpoint e revê uma experiência anterior para reduzir esquecimento.
3. A cada quatro ciclos, avalia a perda sobre exemplos de teste independentes.
4. Se os resultados estagnarem, existir experiência suficiente e houver memória disponível, propõe uma expansão de 16 canais MLP em cada camada.
5. Compara um treino extra **sem expansão** com o **mesmo treino após expansão**. Só aceita o crescimento se os exemplos independentes melhorarem mais com a expansão; caso contrário, reverte o checkpoint.

A melhoria é experimental. Novos neurónios **não tornam automaticamente a IA mais inteligente**. A avaliação usa tarefas muito simples, por isso uma melhoria não demonstra linguagem ou raciocínio geral. O agente nunca treina automaticamente com respostas inventadas pelo próprio Transformer.

O treino de exercícios é limitado, não envolve processos externos e utiliza uma única thread de treino protegida por um lock. Os eventos, incluindo rejeições de crescimento, são guardados em SQLite e aparecem no `/cerebro`.

### Configuração

| Variável | Por defeito | Objetivo |
|---|---|---|
| `NORD_BRAIN_PRESET` | `small` | `tiny`, `small`, `medium` na primeira execução |
| `NORD_AUTO_TRAIN` | `on` | Revisão e treino periódico |
| `NORD_AUTO_EXPERIMENTS` | `on` | Gerar experiências aritméticas verificáveis |
| `NORD_AUTO_GROW` | `on` | Experimentar crescimento condicionado a testes |
| `NORD_AUTO_INTERVAL` | `30` | Segundos entre ciclos (mínimo 5) |
| `NORD_MAX_PARAMS` | `3000000` | Limite máximo de parâmetros |
| `NORD_MAX_MLP_WIDTH` | `1024` | Limite de canais por MLP |
| `NORD_INTERNET` | `on` | Documentação primária selecionada (off para desligar) |
| `NORD_AUTONOMOUS_RESEARCH` | `on` | Pesquisar dúvidas pendentes, mas só quando a Internet está ativa |
| `NORD_SOURCE_TRAIN` | `on` | Permitir treino com excertos da documentação primária |
| `NORD_HYPOTHESES_PER_TICK` | `3` | Experiências de hipóteses matemáticas por ciclo (1–8) |
| `NORD_SHOW_ORIGIN` | `off` | Mostrar origem de cada resposta |
| `NORD_THREADS` | `1` | Threads de CPU do PyTorch |

Exemplo de pesquisa com autonomia, mantendo fonte não verificada:

```bash
NORD_INTERNET=on NORD_AUTONOMOUS_RESEARCH=on NORD_BRAIN_PRESET=tiny .venv/bin/python -m nord_brain
```

A pesquisa agora consulta somente páginas oficiais explicitamente autorizadas de Python, Raspberry Pi, NASA e Linux. Pequenos excertos de origem identificada podem treinar o modelo, mas não são automaticamente confirmados como verdade científica. Consultas fora desta lista não são feitas.

## Estrutura

```text
nord_brain/
  core.py          # Transformer, gradientes, avaliação e MLP expansível
  runtime.py       # Conversa, ensino, memória, acesso controlado ao modelo
  conversation.py  # Reconhecimento transparente de padrões em português
  autonomy.py      # Ciclo autónomo, experiências e ensaios de crescimento
  experiments.py   # Exercícios verificáveis com conjuntos de teste distintos
  reasoning.py     # Provas por relações ensinadas, hipóteses e aritmética
  storage.py       # Memória, lições, testes, eventos, estados da autonomia
  vocabulary.py    # Sementes lexicais iniciais (não pré-treinam)
  internet.py      # Documentação primária com whitelist; sem Wikipédia
  hypotheses.py    # Exploração de hipóteses matemáticas verificadas
  natural_math.py  # Deteta contas e explicações em linguagem corrente
  world.py         # Simulador sensoriomotor treinável em CPU
  cli.py           # Terminal conversacional
```

A memória anterior e o checkpoint V0.2 continuam compatíveis. A principal interface passou a ser conversa, mas `/ajuda`, `/cerebro`, `/avaliar` e `/sair` continuam disponíveis como comandos de diagnóstico. O serviço web antigo não foi restabelecido.

## Verificar

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Os testes cobrem também inferência em várias etapas, hipóteses temporárias, aritmética segura, ensino conversacional e treino derivado. Os restantes testes cobrem causalidade, pesos alterados e persistidos, frases naturais de ensino e correção, janela de treino longa, treino autónomo em thread, separação de treino/teste e decisão reversível de crescimento. **Estes são testes de implementação, não provas de inteligência**.

### Limites reais

- Começar com pesos aleatórios exige **muitos dados, esforço computacional e aprendizagem validada** antes de se obter boa linguagem. Um vocabulário não substitui o pré-treino.
- Autonomia é limitada a ações e verificadores implementados. Pequenos excertos de documentação primária podem treinar os pesos (desativável), mas sem prova independente de veracidade.
- O crescimento é condicionado a **plateau, testes separados e ganho sobre um controlo sem crescimento**, além de limites de memória; pode nunca ser aprovado, o que é desejável se não houver benefício.
- Este programa não gere sozinho o hardware nem é um cérebro comparável ao de uma pessoa. Não corre integralmente num microcontrolador.
