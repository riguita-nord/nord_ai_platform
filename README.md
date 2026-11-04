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

**Limites:** a linguagem compreendida pelo verificador é restrita (`é`, `tem`, `todos os ... são/têm`, expressões aritméticas e alguns problemas simples). Não existe compreensão geral, raciocínio humano nem verificação automática da verdade das afirmações do professor. A expansão de neurónios da V0.3 continua a ser experimental e não depende de deduções sem testes independentes. A Internet permanece separada dos dados de treino.

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

Um agente local executa um ciclo de aprendizagem por defeito a cada 60 segundos, enquanto o processo está aberto:

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
| `NORD_AUTO_INTERVAL` | `60` | Segundos entre ciclos (mínimo 5) |
| `NORD_MAX_PARAMS` | `3000000` | Limite máximo de parâmetros |
| `NORD_MAX_MLP_WIDTH` | `1024` | Limite de canais por MLP |
| `NORD_INTERNET` | `off` | Autorizar pesquisa pública limitada |
| `NORD_AUTONOMOUS_RESEARCH` | `on` | Pesquisar dúvidas pendentes, mas só quando a Internet está ativa |
| `NORD_SHOW_ORIGIN` | `off` | Mostrar origem de cada resposta |
| `NORD_THREADS` | `1` | Threads de CPU do PyTorch |

Exemplo de pesquisa com autonomia, mantendo fonte não verificada:

```bash
NORD_INTERNET=on NORD_AUTONOMOUS_RESEARCH=on NORD_BRAIN_PRESET=tiny .venv/bin/python -m nord_brain
```

A pesquisa usa **apenas a API da Wikipédia em português**: HTTPS, tamanho e tempo limitados, sem URL livre nem redirecionamentos. Os excertos são guardados como observações **não verificadas** e não são transformados automaticamente em exemplos de treino. Acesso livre e treino direto de qualquer página seriam inseguros e podem degradar o modelo. Internet não equivale a ter conhecimento validado.

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
  internet.py      # Pesquisa opcional com fonte
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
- Autonomia é limitada a ações e verificadores implementados, e a leitura da Internet não ensina automaticamente os pesos.
- O crescimento é condicionado a **plateau, testes separados e ganho sobre um controlo sem crescimento**, além de limites de memória; pode nunca ser aprovado, o que é desejável se não houver benefício.
- Este programa não gere sozinho o hardware nem é um cérebro comparável ao de uma pessoa. Não corre integralmente num microcontrolador.
