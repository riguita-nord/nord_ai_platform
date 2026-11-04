"""A starter lexicon, not pretended neural knowledge.

Words are provided as teacher-created seeds. The neural network starts with
random weights, so merely listing words does not teach language comprehension.
"""
from __future__ import annotations
from pathlib import Path
import json

GROUPS = {
    'pronomes': 'eu tu ele ela nós vocês eles elas isto isso aquilo alguém ninguém todos outro próprio meu teu nosso este esse cada algum nenhum quem qual',
    'verbos': 'ser estar ter fazer dizer ir vir dar ver ouvir sentir pensar aprender ensinar perguntar responder conhecer saber querer poder precisar tentar procurar encontrar criar mudar crescer guardar lembrar esquecer entender compreender perceber explicar mostrar ler escrever falar conversar trabalhar jogar andar correr saltar dormir comer beber abrir fechar começar terminar continuar esperar gostar ajudar usar chamar escolher decidir comparar observar experimentar verificar corrigir errar acertar contar somar subtrair dividir multiplicar medir existir acontecer imaginar descobrir reconhecer escolher construir receber enviar pesquisar navegar investigar estudar repetir praticar melhorar',
    'conectores': 'e ou mas porque pois portanto então se quando enquanto embora também ainda já nunca sempre talvez apenas quase antes depois durante até desde com sem para por sobre entre dentro fora acima abaixo perto longe aqui ali lá onde como quanto porque agora depois hoje ontem amanhã',
    'qualidades': 'bom mau grande pequeno alto baixo forte fraco rápido lento fácil difícil novo velho certo errado verdadeiro falso feliz triste quente frio leve pesado simples complexo aberto fechado cheio vazio igual diferente maior menor possível impossível seguro perigoso limpo sujo bonito feio claro escuro importante útil inútil',
    'tempo': 'segundo minuto hora dia semana mês ano manhã tarde noite passado presente futuro início fim tempo momento idade cedo tarde ontem hoje amanhã',
    'numeros': 'zero um dois três quatro cinco seis sete oito nove dez onze doze treze catorze quinze dezasseis dezassete dezoito dezanove vinte trinta quarenta cinquenta cem mil milhão primeiro segundo terceiro metade dobro',
    'pessoas': 'pessoa criança bebé adulto amigo amiga família mãe pai filho filha irmão irmã avô avó professor professora aluno aluna homem mulher humano grupo equipa comunidade',
    'corpo': 'cabeça mão braço perna pé olho ouvido nariz boca coração cérebro corpo pele sangue osso cabelo rosto dedo voz',
    'natureza': 'água ar terra fogo pedra areia árvore planta flor folha raiz fruto semente sol lua estrela céu nuvem chuva vento neve mar rio lago oceano montanha floresta campo natureza animal cão gato pássaro peixe inseto',
    'objetos': 'casa porta janela mesa cadeira cama quarto cozinha livro papel lápis caneta copo prato caixa bola brinquedo roupa sapato mochila chave carro bicicleta estrada ponte cidade aldeia escola loja hospital',
    'tecnologia': 'computador teclado rato ecrã terminal sistema programa código ficheiro pasta rede internet página sítio mensagem texto imagem som vídeo dado número palavra frase pergunta resposta erro teste memória modelo neurónio peso ligação inteligência aprendizagem informação conhecimento ideia conceito ferramenta comando sensor processo processador energia bateria servidor placa aplicação máquina robô',
    'abstratos': 'vida mundo coisa objeto espaço movimento posição distância direção forma cor tamanho quantidade valor razão causa consequência efeito motivo objetivo plano decisão hipótese experiência exemplo problema solução verdade dúvida certeza regra padrão diferença semelhança relação parte conjunto ordem sequência escolha liberdade limite risco segurança curiosidade capacidade competência compreensão conhecimento raciocínio lógica matemática ciência linguagem palavra significado nome',
    'cores': 'vermelho verde azul amarelo laranja roxo rosa preto branco cinzento castanho dourado prateado',
    'saudacoes': 'olá adeus obrigado obrigada desculpa por favor bem-vindo bom dia boa tarde boa noite sim não'
}


def seeds():
    return {word: category for category, words in GROUPS.items() for word in words.split()}


def initialize(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(seeds(),ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return json.loads(path.read_text(encoding='utf-8'))
