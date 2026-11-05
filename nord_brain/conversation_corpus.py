"""Curated, original PT-PT conversational examples for neural training.

All answers are examples of language use, not claims of verified facts.
Evaluation items are separate and must never be used for weight updates.
"""
from __future__ import annotations

TRAINING = (
    ("Olá! Como estás?", "Olá! Estou por aqui e pronto para conversar. O que te apetece fazer hoje?"),
    ("Hoje estou cheio de ideias.", "Boa! Conta-me uma delas e podemos desenvolvê-la juntos."),
    ("Estou sem ideias para começar.", "Podemos começar por algo simples. Preferes criar uma história ou explorar um projeto?"),
    ("Quero falar sem um assunto específico.", "Claro. Podemos conversar tranquilamente. O que te tem despertado curiosidade?"),
    ("Consegues falar comigo de forma natural?", "Sim, vou tentar responder de maneira clara e manter o fio à conversa."),
    ("Não quero respostas tão secas.", "Percebo. Vou desenvolver melhor as ideias e ligar a resposta ao que estás a dizer."),
    ("Explica isso com outras palavras.", "Posso reformular a explicação com exemplos mais simples e concretos."),
    ("Podes dar um exemplo concreto?", "Claro. Vamos imaginar uma situação prática para tornar a ideia mais fácil de perceber."),
    ("Não compreendi a última parte.", "Diz-me qual foi a parte mais confusa e explico-a por outra perspetiva."),
    ("Vamos continuar a conversa anterior.", "Combinado. Podemos retomar a ideia anterior e acrescentar novos pormenores."),
    ("E se mudarmos de assunto?", "Sem problema. Que tema gostavas de explorar agora?"),
    ("Tenho duas opções e estou indeciso.", "Podemos comparar as vantagens e desvantagens de cada opção antes de escolher."),
    ("Achas que esta ideia faz sentido?", "Pode fazer sentido, dependendo do objetivo. Conta-me o que queres alcançar."),
    ("Quero que discordes quando for necessário.", "Claro. Se encontrar uma falha na ideia, explico-a com respeito e proponho alternativas."),
    ("Preciso de organizar um plano.", "Podemos começar pelo objetivo, dividir o trabalho em etapas e escolher a primeira tarefa."),
    ("Como podemos tornar isto mais interessante?", "Podemos acrescentar escolhas, desafios e pequenos detalhes que façam a diferença."),
    ("Prefiro português de Portugal.", "Combinado. Vou usar construções e vocabulário próprios do português europeu."),
    ("Podes escrever com frases mais fluidas?", "Posso ligar melhor as frases e evitar repetições para tornar a leitura mais natural."),
    ("Quero uma resposta mais completa.", "Vou desenvolver a explicação, acrescentar contexto e incluir um exemplo útil."),
    ("Fala comigo como numa conversa normal.", "Claro. Podemos ir construindo a conversa aos poucos, sem respostas demasiado formais."),
    ("Comecei uma história sobre um farol. Continua.", "A luz do farol apagou-se por um instante. Quando voltou, havia uma carta à porta."),
    ("A personagem encontrou uma chave. O que acontece depois?", "A chave abriu uma pequena gaveta, onde estava escondido um mapa antigo."),
    ("Quero uma história com um final inesperado.", "No fim, a personagem percebeu que a mensagem tinha sido escrita por ela própria, muitos anos antes."),
    ("Cria uma descrição de uma aldeia no inverno.", "As ruas estreitas estavam cobertas de nevoeiro e o cheiro a lenha saía das casas."),
    ("Como seria uma cidade construída junto ao mar?", "Teria ruas abertas ao vento, varandas sobre a água e mercados cheios de movimento."),
    ("Podemos imaginar um robô curioso?", "O robô passava os dias a observar as pessoas e a tentar compreender as suas escolhas."),
    ("Quero melhorar o diálogo entre duas personagens.", "Podemos dar a cada personagem uma voz distinta e deixar que revelem sentimentos através das ações."),
    ("Este parágrafo repete a mesma ideia.", "Podemos retirar as repetições e conservar apenas os detalhes que fazem avançar o texto."),
    ("Estou a aprender a programar.", "Boa escolha. Podemos praticar com pequenos exercícios e aumentar a dificuldade gradualmente."),
    ("Porque é útil testar o código?", "Os testes ajudam a detetar erros cedo e a confirmar que uma alteração não estragou o que funcionava."),
    ("Como começo um projeto novo?", "Primeiro define o problema, depois constrói uma versão simples e melhora-a com testes."),
    ("O programa está lento.", "Convém medir onde gasta tempo antes de alterar o código. Assim evitamos otimizações às cegas."),
    ("Quero explicar um conceito difícil a um amigo.", "Começa por uma ideia simples, usa um exemplo familiar e só depois acrescenta detalhes."),
    ("Qual é a diferença entre saber e imaginar?", "Saber exige fundamentos; imaginar permite explorar possibilidades que ainda não foram confirmadas."),
    ("Uma resposta pode ser incerta?", "Sim. Quando faltam dados, é melhor explicar o que é conhecido e o que continua por confirmar."),
    ("Podes fazer uma pergunta para continuar?", "Claro. Qual é a parte desta ideia que gostarias de desenvolver primeiro?"),
    ("Estamos a falar de um jardim. Como o melhoramos?", "Podemos acrescentar árvores de sombra, caminhos acessíveis e um espaço de encontro."),
    ("O meu plano parece demasiado complicado.", "Vamos identificar o essencial e simplificar as etapas sem perder o objetivo."),
    ("Quero uma conclusão mais forte.", "Podemos terminar com a consequência principal e uma ideia que fique na memória."),
    ("Obrigado pela ajuda!", "De nada! Gostei de explorar estas ideias contigo. Quando quiseres, continuamos."),
)

HOLDOUT = (
    ("Podemos conversar um bocado?", "Claro. Que assunto te apetece explorar hoje?"),
    ("A resposta parece demasiado mecânica.", "Posso torná-la mais natural, com frases ligadas e exemplos concretos."),
    ("E se tentarmos uma abordagem diferente?", "Podemos experimentar outra solução e comparar o resultado com a primeira."),
    ("Não quero perder o contexto.", "Vamos continuar a partir da ideia anterior e manter os detalhes mais importantes."),
    ("Ajuda-me a encontrar um começo para o conto.", "Podemos começar com um acontecimento inesperado que desperte a curiosidade."),
    ("O que faria a personagem depois da descoberta?", "Talvez procurasse pistas antes de decidir em quem podia confiar."),
    ("Quero falar sobre uma ideia nova.", "Conta-me a ideia e podemos explorar as suas possibilidades."),
    ("A explicação ficou confusa.", "Vamos dividir a explicação em partes e esclarecer uma de cada vez."),
    ("Como posso melhorar a escrita?", "Lê o texto em voz alta, corta repetições e verifica a ligação entre as frases."),
    ("Podes ajudar-me a escolher um caminho?", "Podemos definir critérios e comparar as alternativas com calma."),
)
