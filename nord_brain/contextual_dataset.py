"""Original PT-PT multi-turn curriculum for contextual interpretation.

Train and holdout are separate; never use holdout for weight updates.
The same surface expression deliberately has different meanings in context.
"""
from __future__ import annotations
from .dialogue_format import training_prompt

TRAIN_DIALOGUES = (
    ((('user','Sobre o que queres conversar?'),('assistant','Escolhe um assunto.'),('user','Qualquer coisa.')), 'Então vamos imaginar uma viagem ao futuro. Como achas que serão as cidades?'),
    ((('user','O que comeste ao almoço?'),('assistant','Estou curioso: o que foi?'),('user','Uma coisa qualquer.')), 'Percebo, não foi nada de especial. Apetece-te falar de outra coisa?'),
    ((('user','Quero falar de música.'),('assistant','Qual é o teu género preferido?'),('user','Afinal, escolhe um tema aleatório.')), 'Vamos mudar de assunto: que tecnologia gostavas de ver inventada?'),
    ((('user','Tenho uma ideia para um jogo.'),('assistant','Conta-me a ideia.'),('user','Não sei bem como explicar.')), 'Podes começar pelo objetivo do jogador, mesmo que ainda não tenhas todos os detalhes.'),
    ((('user','Gosto de explorar cidades.'),('assistant','Preferes cidades antigas ou modernas?'),('user','Tanto faz.')), 'Então imagina uma cidade antiga com transportes futuristas. O que gostavas de visitar primeiro?'),
    ((('user','Estou a aprender programação.'),('assistant','Queres começar com exemplos simples?'),('user','Prefiro um desafio.')), 'Podemos criar um pequeno programa que organiza tarefas por prioridade.'),
    ((('user','Quero falar de viagens.'),('assistant','Qual é o teu destino favorito?'),('user','Esquece, vamos falar de livros.')), 'Claro, mudamos para livros. Preferes ficção ou não ficção?'),
    ((('user','Ontem vi um banco no jardim.'),('assistant','Era de madeira?'),('user','Sim, estava junto ao lago.')), 'Parece um sítio tranquilo para descansar e observar a paisagem.'),
    ((('user','Preciso de ir ao banco.'),('assistant','É para tratar de dinheiro?'),('user','Sim, tenho de abrir uma conta.')), 'Nesse caso, vale a pena comparar as condições das contas disponíveis.'),
    ((('user','Quero criar uma história.'),('assistant','Sobre o quê?'),('user','Surpreende-me.')), 'Uma manhã, todas as sombras da cidade começaram a apontar para o mesmo lugar.'),
    ((('user','Hoje não estou com disposição.'),('assistant','Queres falar sobre isso?'),('user','Não, obrigado.')), 'Está bem. Podemos ficar por aqui ou mudar para algo mais leve.'),
    ((('user','Podes explicar redes neuronais?'),('assistant','Claro, queres uma explicação simples?'),('user','Mais técnica.')), 'Uma rede neuronal ajusta parâmetros para reduzir uma função de perda durante o treino.'),
    ((('user','Olá!'),('assistant','Olá, como estás?'),('user','Estou bem.'),('assistant','Ainda bem! Como foi o teu dia?'),('user','Hoje tive programação e correu muito mal.')), 'Que chatice. O que correu mal na aula de programação?'),
    ((('user','Como foi o teu dia?'),('assistant','Conta-me como correu.'),('user','Tive um teste difícil e fiquei frustrado.')), 'Imagino que tenha sido frustrante. Qual foi a parte mais difícil do teste?'),
    ((('user','Queres contar-me alguma coisa?'),('assistant','Estou a ouvir.'),('user','Ontem fui passear e encontrei um amigo antigo.')), 'Que coincidência! Há quanto tempo não o vias?'),
    ((('user','Hoje estive a programar.'),('assistant','O que estavas a fazer?'),('user','Tentei corrigir um erro mas não consegui.')), 'Qual era o erro? Podemos tentar perceber juntos o que aconteceu.'),
    ((('user','Queres falar do teu dia?'),('assistant','Sim, conta-me.'),('user','Tive uma aula de matemática excelente.')), 'Boa! O que aprendeste nessa aula?'),
)
HOLDOUT_DIALOGUES = (
    ((('user','Queres falar de filmes ou jogos?'),('assistant','Podes escolher.'),('user','Escolhe tu.')), 'Vamos falar de jogos: que mecânica gostavas de inventar?'),
    ((('user','O que compraste?'),('assistant','Conta-me o que encontraste.'),('user','Uma coisa qualquer.')), 'Percebi, não queres especificar o que compraste.'),
    ((('user','Estamos a falar de computadores.'),('assistant','Queres comparar processadores?'),('user','Agora quero falar de natureza.')), 'Claro, mudamos para natureza. Preferes montanhas ou mar?'),
    ((('user','Vi um banco na praça.'),('assistant','Estavas a passear?'),('user','Sentei-me nele.')), 'Então era um banco para descansar na praça.'),
)

def examples(context: int, *, holdout: bool=False):
    source=HOLDOUT_DIALOGUES if holdout else TRAIN_DIALOGUES
    return tuple((training_prompt(list(turns), context), answer) for turns,answer in source)
