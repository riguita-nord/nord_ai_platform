"""Terminal conversation first. Diagnostic commands are optional."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .runtime import Brain

HELP='''Fala normalmente comigo. Exemplos:
  Olá
  Chamo-me Ricardo.
  O Python é uma linguagem de programação.
  O que é Python?
  Quando eu perguntar como te chamas?, responde Nord Brain.
  Não, a resposta correta é outra.
  O que aprendeste?
  Pesquisa sobre vulcões. (se a Internet estiver ativa)

Comandos técnicos opcionais: /cerebro, /avaliar, /ajuda, /sair
A aprendizagem, a revisão e a decisão experimental de crescimento funcionam
sem comandos, conforme os limites e testes configurados. A rede começa sem
pré-treino, por isso a linguagem e o raciocínio ainda serão muito limitados.
'''


def handle(brain,line):
    # Legacy debugging interface; teaching no longer requires these commands.
    if line.startswith('/ensinar '):
        parts=line[len('/ensinar '):].split('=>',1)
        if len(parts)!=2:raise ValueError('Usa pergunta => resposta')
        loss=brain.teach(parts[0],parts[1])
        return f'Exemplo treinado nos pesos, loss={loss:.4f}'
    if line.startswith('/teste '):
        parts=line[len('/teste '):].split('=>',1)
        if len(parts)!=2:raise ValueError('Usa pergunta => resposta')
        with brain.lock:brain.memory.holdout(parts[0].strip(),parts[1].strip())
        return 'Teste independente guardado.'
    if line=='/ajuda':return HELP
    if line=='/cerebro':return json.dumps(brain.status(),ensure_ascii=False,indent=2)
    if line=='/avaliar':
        result=brain.evaluate()
        return 'Ainda não há testes independentes.' if result is None else f'Acertos neuronais em problemas reservados: {result[0]}/{result[1]}'
    if line=='/sair':return None
    return 'Comando desconhecido. Experimenta falar normalmente ou usa /ajuda.'


def main(argv=None):
    parser=argparse.ArgumentParser(description='Nord Brain: conversa natural e aprendizagem autónoma experimental')
    parser.add_argument('--data',type=Path,default=Path(os.environ.get('NORD_DATA',str(Path.home()/'.nord-brain'))))
    args=parser.parse_args(argv)
    brain=Brain(args.data)
    print('Nord Brain v0.4 | Transformer próprio | aprendizagem autónoma experimental')
    print('Fala normalmente. Escreve /ajuda para exemplos ou /sair para terminar.')
    try:
        while True:
            try:line=input('tu> ').strip()
            except EOFError:break
            if not line:continue
            if line.lower() in ('/sair','sair','exit','quit'):break
            try:
                if line.startswith('/'):
                    response=handle(brain,line)
                    if response is None:break
                    print(response)
                else:
                    response,origin=brain.chat(line)
                    print('nord> '+response)
                    if os.getenv('NORD_SHOW_ORIGIN','off').lower() in ('on','1','true'):
                        print(f'  [origem: {origin}]')
            except (ValueError,OverflowError,ConnectionError,TimeoutError,OSError) as error:
                print('nord> Não consegui completar isso:',error)
    except KeyboardInterrupt:
        print('\nA terminar...')
    finally:
        brain.close()
        print('Estado do cérebro guardado.')


if __name__=='__main__':main()
