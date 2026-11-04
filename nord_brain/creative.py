"""Deterministic structure with randomized storytelling as *offline fallback*.

This is not a replacement for a pretrained creative language model. It ensures
requests are handled even when the optional local llama.cpp backend is absent,
without storing a story prompt as a false teaching example.
"""
from __future__ import annotations
import random
import re
import unicodedata


def _normal(value: str) -> str:
    s=unicodedata.normalize('NFKD',value.casefold())
    return ''.join(c for c in s if not unicodedata.combining(c))


def creative_request(message: str) -> str | None:
    text=' '.join(message.split()).strip('?.! ')
    norm=_normal(text)
    # Not every usage of "história" is fiction: distinguish factual research.
    if re.match(r'^(?:qual e a historia de|conta me a historia real de|quando aconteceu)',norm):
        return None
    match=re.fullmatch(
        r'(?:conta(?:-me| me)?|inventa|cria|escreve|faz)\s+'
        r'(?:(?:uma|um)\s+)?(?:historia|hitoria|estoria|conto|narrativa|fabula|aventura)'
        r'(?:\s+(?:sobre|acerca de|com)\s+(.+))?',norm)
    if match:return (match.group(1) or 'uma descoberta inesperada')[:90]
    match=re.fullmatch(r'(?:da(?:-me| me)?|sugere|inventa|cria)\s+(?:umas?\s+)?ideias?\s+(?:para|sobre)\s+(.+)',norm)
    if match:return 'IDEIAS:'+match.group(1)[:90]
    return None


CHARACTERS=('Inês, uma jovem inventora','Tomás, um velho cartógrafo',
            'Lia, uma aprendiz de astronomia','Rui, um reparador de autómatos',
            'Mara, uma bibliotecária curiosa','Nuno, um músico que colecionava enigmas')
PLACES=('numa aldeia junto ao mar','numa cidade escondida entre montanhas',
        'numa estação abandonada','num laboratório na periferia da cidade',
        'numa ilha que não constava de nenhum mapa')
DISCOVERIES=('um caderno capaz de registar perguntas ainda não feitas',
             'uma máquina que só funcionava quando alguém admitia estar errado',
             'uma carta escrita por uma pessoa que ainda não tinha nascido',
             'uma chave que abria apenas portas esquecidas',
             'uma bússola que apontava para coisas perdidas')
CONFLICTS=('ninguém acreditava na descoberta e o tempo começava a esgotar-se',
           'cada tentativa de resolver o enigma fazia desaparecer uma memória',
           'a única solução exigia confiar numa antiga rival',
           'a descoberta ameaçava revelar um segredo que protegia a comunidade')
CHOICES=('partilhar o segredo com quem mais precisava de ajuda',
         'desmontar a invenção e reconstruí-la com outros princípios',
         'trocar a certeza de uma resposta pela coragem de fazer uma nova pergunta',
         'pedir ajuda a alguém com uma perspetiva completamente diferente')


def story(theme: str, *, rng: random.Random | None=None) -> str:
    rng=rng or random.SystemRandom()
    person=rng.choice(CHARACTERS)
    place=rng.choice(PLACES)
    found=rng.choice(DISCOVERIES)
    conflict=rng.choice(CONFLICTS)
    choice=rng.choice(CHOICES)
    clean=re.sub(r'[\x00-\x1f<>]',' ',theme).strip()[:85]
    if clean.startswith('IDEIAS:'):
        topic=clean[7:]
        return (f'Para {topic}, exploraria três possibilidades:\n'
                f'1. Adaptar uma ideia de outro domínio: {found}.\n'
                f'2. Começar por um protótipo simples {place} e observar os erros.\n'
                f'3. Envolver pessoas com experiências diferentes e comparar resultados.\n'
                'Escolheria primeiro um pequeno teste e avaliaria custo, risco e utilidade. '
                'Estas são sugestões experimentais, não resultados comprovados.')
    title=rng.choice(('A pergunta esquecida','O mapa do impossível','A última experiência',
                      'O segredo da oficina','O dia em que tudo mudou'))
    return (f'{title}\n\n'
        f'{person} vivia {place}. Desde pequena, havia uma ideia que lhe tirava o sono: {clean}. '
        f'Numa manhã de chuva, encontrou {found}. Não parecia importante, até perceber que '
        'o objeto reagia de forma diferente a cada pergunta.\n\n'
        f'Quando tentou explicar o que descobrira, percebeu que {conflict}. '
        'Passou a noite a comparar hipóteses, a registar erros e a experimentar alternativas. '
        'Algumas falharam; outras mostraram-lhe que o problema não era bem aquele que imaginara.\n\n'
        f'Por fim, decidiu {choice}. Não resolveu todos os mistérios, mas conseguiu '
        'dar o primeiro passo sem esconder as suas dúvidas. Quando amanheceu, '
        'a cidade continuava igual — e, no entanto, para ela, o mundo já era maior.')
