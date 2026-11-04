"""Contextual narrative revisions for the offline story fallback.

A bounded procedural revision, explicitly not neural creativity. It preserves
names and the previously generated events when no local language model runs.
"""
from __future__ import annotations

import re


def _elements(original: str) -> tuple[str, str, str]:
    first = original.split('\n\n', 1)[-1]
    person = re.search(r'([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ-]+),\s+uma?\s+', first)
    name = person.group(1) if person else 'a protagonista'
    location = re.search(r'vivia\s+(.+?)\.', first, re.I | re.S)
    where = location.group(1).strip()[:90] if location else 'naquele lugar'
    discovery = re.search(r'encontrou\s+(.+?)\.', first, re.I | re.S)
    object_name = discovery.group(1).strip()[:120] if discovery else 'a descoberta'
    return name,where,object_name


def revise_story(original: str, kind: str = 'longer') -> str:
    """Continue or expand a prior story, retaining the original as the backbone."""
    original = original.strip()
    if not original:
        return ''
    name,where,discovery = _elements(original)
    paragraphs = [p.strip() for p in original.split('\n\n') if p.strip()]
    if kind == 'continue':
        return ('Um novo capítulo\n\n'
                f'Na manhã seguinte, {name} regressou ao lugar onde tudo começara, {where}. '
                f'A experiência com {discovery} não lhe saía da cabeça. Era possível que '
                'a decisão tomada tivesse resolvido apenas uma parte do mistério.\n\n'
                f'Para descobrir, {name} começou por escrever o que tinha observado e o '
                'que continuava a ser apenas uma suspeita. Ao comparar as duas listas, '
                'notou uma contradição: uma pista só podia ser verdadeira se outra fosse falsa.\n\n'
                '— Talvez não estejamos a fazer a pergunta certa — disse em voz baixa.\n'
                'Decidiu regressar à primeira pista e pedir a alguém que observasse os '
                'mesmos acontecimentos sem conhecer as conclusões anteriores.\n\n'
                f'O reencontro trouxe uma surpresa: {discovery} escondia uma marca que '
                'passara despercebida. A nova descoberta não encerrava a história; '
                f'dava a {name} uma razão concreta para continuar a investigar.')
    if kind == 'ending':
        return ('\n\n'.join(paragraphs[:-1])+'\n\n'
                f'No momento decisivo, {name} percebeu que não precisava de uma resposta '
                f'perfeita sobre {discovery}. Precisava de partilhar as provas com quem '
                'seria afetado pela escolha. Quando explicou o que sabia e o que ainda '
                'ignorava, encontrou ajuda onde menos esperava.\n\n'
                'O mistério não desapareceu por magia. Foi resolvido com uma decisão '
                'difícil, várias tentativas e a coragem de admitir os próprios erros. '
                'Ao regressar a casa, sabia que o dia seguinte não seria igual aos anteriores.')
    if kind == 'rewrite':
        # Provide a different structure while retaining established facts.
        kind = 'longer'
    scene_a = (
        f'Antes de avançar, {name} percorreu devagar os corredores {where}. '
        'O ar tinha o cheiro das coisas esquecidas e cada som parecia ficar suspenso '
        'por um instante. Reparou que pequenos detalhes, antes irrelevantes, '
        'formavam agora uma sequência: um sinal junto à porta, uma anotação no chão '
        'e um silêncio estranho quando o assunto era mencionado. Não sabia ainda '
        'o que significavam, por isso evitou tirar conclusões precipitadas.\n\n'
        f'Naquela noite, {name} colocou {discovery} sobre a mesa e decidiu começar '
        'do princípio. Primeiro registou as circunstâncias da descoberta; depois '
        'tentou repetir o que acontecera. Uma tentativa não produziu qualquer '
        'resultado. A seguinte revelou um pormenor novo. A diferença era tão '
        'pequena que quase passou despercebida, mas mostrava que havia uma regra '
        'por descobrir.')
    scene_b = (
        f'{name} procurou então alguém disposto a ouvir a história inteira, '
        'sem escolher logo uma explicação. A conversa foi difícil: quem a escutou '
        'fez perguntas que pareciam óbvias e que, mesmo assim, nunca tinham sido '
        'consideradas. Algumas hipóteses foram abandonadas. Outras ficaram anotadas '
        'para serem testadas no dia seguinte.\n\n'
        f'Quando regressou {where}, {name} observou {discovery} sob uma luz diferente. '
        'O objeto continuava o mesmo, mas a maneira de olhar para ele tinha mudado. '
        'Havia um caminho que parecia promissor e outro que oferecia uma saída fácil '
        'demais. Escolher o primeiro significava correr o risco de descobrir que '
        'a ideia inicial estava errada. Ainda assim, decidiu experimentar.')
    scene_c = (
        'A experiência não correu como esperava. A primeira pista conduziu a um '
        'beco sem saída e uma pequena falha obrigou todos a recomeçar. Em vez de '
        'esconder o erro, decidiram estudá-lo. Só então perceberam a ligação entre '
        'as pistas antigas e o acontecimento mais recente.\n\n'
        f'{name} sentiu alívio, mas também receio: compreender uma parte do enigma '
        'criava uma responsabilidade nova. As pessoas à sua volta não precisavam '
        'apenas de uma descoberta impressionante. Precisavam de saber o que ela '
        'mudaria nas suas vidas. A decisão final teria de ter isso em conta.')
    if len(paragraphs) >= 3:
        return '\n\n'.join(paragraphs[:2]+[scene_a,paragraphs[2],scene_b,scene_c]+paragraphs[3:])
    return original+'\n\n'+scene_a+'\n\n'+scene_b+'\n\n'+scene_c
