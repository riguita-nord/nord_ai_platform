# Ambiente de desenvolvimento e validação do Transformer Nord

## Ambiente utilizado

- Windows x64, Python 3.12.10 (instalado e disponibilizado pelo launcher `py`; não alterado).
- Ambiente virtual local: `.venv` (ignorado pelo Git).
- PyTorch 2.15.0+cpu; `torch.cuda.is_available()` foi `False`.
- NumPy 2.5.4.
- Dependências instaladas a partir de `requirements.txt`; pip atualizado dentro de `.venv` para 26.3.
- Não foi encontrado `~/.nord-brain/brain-v2.pt`; nenhum checkpoint do utilizador foi aberto ou alterado. Os testes e a experiência usam diretórios temporários.

## Preparação e comandos reproduzidos

PowerShell (Windows):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_brain.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_generalization.py -v
.\.venv\Scripts\python.exe scripts\experiment_plural_generalization.py
'/sair' | .\.venv\Scripts\python.exe -m nord_brain --data "$env:TEMP\nord-cli-smoke"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

A imagem Windows instala a build CPU do PyTorch sem CUDA. Para correr o launcher `./nord` e validar a instalação Ubuntu/Linux, usar a VM (sem partilhar o diretório de dados com outro processo):

```bash
cd ~/nord_ai_platform
git status --short --branch
python3 --version
./nord --help
./nord --data /tmp/nord-cli-smoke
# escrever /sair no prompt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/experiment_plural_generalization.py
```

O launcher instala as dependências em `.venv` se necessário. Para confirmar a execução direta do entry point em Linux, enviar `/sair` ao processo `./nord --data /tmp/nord-cli-smoke`.

## Correção de compatibilidade

`nord_brain/cli.py` importava `fcntl` sem alternativa, impedindo importar a CLI em Windows. Foi adicionado bloqueio exclusivo equivalente com `msvcrt` no Windows e mantido `flock` no Linux. A CLI foi iniciada em Windows com um diretório temporário, recebeu `/sair` e terminou guardando o estado. A compatibilidade do launcher `./nord` é específica de Linux e não foi executada neste host Windows.

## Resultados experimentais

### Transformer e persistência

`test_brain.py`: 6/6 passaram. Cobrem atenção causal, atualização de pesos, crescimento preservador de logits, máscara para supervisionar apenas a resposta, treino repetido e persistência/recarga de modelo e estado do otimizador.

### Generalização e separação de dados

`test_generalization.py`: 5/5 passaram. Incluem separação determinística de treino/holdout, ausência das perguntas holdout na formação, verificação de respostas matemáticas e atualização real do modelo.

`scripts/experiment_plural_generalization.py` é uma experiência curta e reproduzível com seed 31, quatro substantivos de treino e dois substantivos reservados. Os prompts de avaliação também são reformulados. Resultado nesta máquina:

- 48 atualizações; 20 tensores de parâmetros foram alterados na execução exploratória inicial.
- Loss média dos bytes-resposta reservados: 5,5946 antes, 4,0222 depois.
- Geração greedy bruta: `ssssssssssssssssssss` para bola e janela.
- Exact match em palavras reservadas: 0/2.
- O checkpoint temporário foi recarregado com 48 passos.

A perda menor mede adaptação da distribuição de bytes, não aprendizagem da regra de plural. A experiência falha no critério de generalização; não se reivindica capacidade de pluralização.

### Regressão completa

`python -m unittest discover -s tests -v`: 175 testes, 42 falhas e 16 erros em 114,5 segundos. A suíte focada no Transformer e a suíte de generalização passaram; a suíte geral não passou.

As falhas restantes abrangem expectativa funcional de conversação, contexto/livros, raciocínio e controlo de aprendizagem. Há também oito erros nos testes do instalador Bash, que tentam executar scripts POSIX em Windows, e erros em testes que falham antes de fechar a base SQLite, levando a falha adicional na limpeza de diretórios temporários Windows. O teste de política de expansão espera uma configuração pequena, mas recebeu o perfil padrão com MLP de largura 384. Estes resultados requerem investigação própria e não são apresentados como regressões resolvidas.

## Achados do repositório que requerem trabalho

- O Transformer canónico está em `nord_brain/core.py`, com bytes UTF-8, EOS/BOS, GQA, RoPE, SwiGLU, pesos partilhados e Autograd.
- A CLI mantém `brain-v2.pt`, mas `Brain` em `nord_brain/runtime.py` cria e usa também `language-v1.pt` e `language_core`. Portanto a aplicação não tem hoje um único checkpoint/motor neuronal para todas as tarefas.
- `scripts/train_nord.py` usa `NeuralCore`, mas permanecem scripts antigos de modelos/currículos independentes, incluindo dependências de `train_language_v2.py` que já não aparece na árvore listada. A unificação pedida no relatório maior não está concluída.
- O runtime expande o Transformer para oito camadas e muda o contexto para 512 ao arrancar. O checkpoint real do utilizador não existia neste ambiente, por isso arquitetura, estado do otimizador e migrações reais não puderam ser inspecionados.
- A representação por byte é segura para acentos UTF-8 e compatível com os checkpoints, mas aumenta a sequência por palavra. A experiência acima demonstra que a capacidade de adaptação observada não implica morfologia generalizada.

## Próximos passos na VM Ubuntu

1. Atualizar a branch `main` e confirmar `git status` antes do treino.
2. Instalar/validar a `.venv` e executar os dois grupos de teste focados e a suíte completa.
3. Executar `./nord --help` e um ciclo `/sair` usando diretório de dados temporário.
4. Antes de qualquer treino real, identificar `brain-v2.pt`/`language-v1.pt`, verificar arquitetura e estado do otimizador e criar cópias. Não usar `--reset` nem executar treino longo até resolver a duplicidade de motores e os holdouts falhados.

## Sessão de conversa e treino no modelo padrão (2026-11-06)

O diretório pessoal `.nord-brain` estava ausente antes desta sessão. A primeira execução criou checkpoints novos, com 0 passos. A aplicação consulta `language-v1.pt` para geração conversacional; `brain-v2.pt` é usado noutros fluxos. Foram criadas cópias antes do treino: `brain-v2.pre-basics.pt`, `language-v1.pre-basics.pt` e `language-v1.after-48.pt`.

A pergunta inicial “Explica o que é uma árvore” recebeu “Ainda não sei responder”, origem `sem treino suficiente`; a geração bruta veio vazia. Foram executadas duas sessões de 48 passos pelo treinador linguístico unificado, usando `training/nord_brain_treino_pt_pt.txt`. Ambas foram aceites pelos holdouts. O checkpoint conversacional atual tem 96 passos; a loss do holdout principal passou de 5,5281 para 4,1909. Isto não se traduziu em conversa coerente: na chamada normal, o modelo produziu “e” para uma pergunta e foi rejeitado na outra.

O gate de candidatos aceitava uma palavra funcional de um carácter (`e`) como resposta. `nord_brain/candidate_evaluation.py` passou a rejeitar palavras únicas com menos de dois caracteres, e foi acrescentada uma regressão em `tests/test_candidate_evaluation.py`. Os testes do ficheiro foram executados diretamente e passaram. Depois da correção, o fragmento deixou de ser apresentado; a pergunta factual foi encaminhada para pesquisa e não encontrou fontes, e a pergunta linguística continuou a ser recusada como saída neuronal não fiável.

Experiências de diagnóstico foram executadas apenas em cópias temporárias:

- Um exemplo de árvore conseguiu ser reproduzido exatamente após 200 atualizações dedicadas, mas a loss dos holdouts linguísticos piorou de 4,1909 para 5,1155. Não foi aplicado ao checkpoint persistente.
- Quatro respostas em blocos (800 atualizações) pioraram o holdout linguístico de 4,1909 para 6,3746 e só 1/4 saídas conhecidas ficou correta. Não foi aplicado.
- Quatro respostas intercaladas com revisão e ancoragem (400 atualizações) melhoraram o holdout principal de 4,1909 para 3,2697, mas as quatro saídas conhecidas e as três perguntas novas continuaram incoerentes. Também não foi aplicado.

Conclusão experimental: o Transformer original consegue memorizar uma resposta através dos seus pesos, mas o treino atual por bytes e teacher forcing não produz geração autoregressiva robusta nem generalização suficiente. A perda de validação, isoladamente, não é um critério adequado para aceitar capacidade conversacional. O checkpoint persistente ficou no estado validado de 96 passos; os testes de memorização regressivos não foram promovidos.

## Experiência adicional de cobertura conversacional (cópia temporária)

Foi testada uma hipótese com 12 perguntas/respostas básicas e conversacionais, incluindo paráfrases no treino, revisão intercalada do corpus e ancoragem dos pesos. Foram 1.400 atualizações adicionais sobre uma cópia do checkpoint com 96 passos.

- As 12 perguntas treinadas geraram exatamente as 12 respostas pretendidas (12/12).
- Nas cinco perguntas reservadas e reformuladas, exact match foi 0/5 e o filtro rejeitou 0/5 gerações como respostas aceitáveis.
- As losses de holdout baixaram: linguagem 4,1909→2,9412; conversa natural 4,1419→2,4680; contexto 4,1904→2,7092; português 4,2584→3,5371.

O treino aprendeu os exemplos específicos, mas não passou o teste de generalização por geração. Por isso, esses pesos não foram promovidos. O checkpoint persistente continua em 96 passos. A experiência confirma que, com o corpus local de 25 KB, baixar a loss ou acertar exemplos conhecidos não basta para obter respostas gerais coerentes.
