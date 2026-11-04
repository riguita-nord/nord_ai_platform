"""Terminal-first interface for teaching and testing a small neural brain."""
import argparse
import json
import os
from pathlib import Path
from .runtime import Brain
from .internet import search

HELP='''Comandos:
  /ensinar pergunta => resposta  Ensina e atualiza realmente os pesos
  /corrigir resposta           Corrige a última pergunta
  /perguntar texto             Responde com memória ou rede neuronal
  /neuronal texto              Testa apenas a rede neuronal (sem memória)
  /conceito palavra = definição  Guarda conceito inicial explicado pelo professor
  /teste pergunta => resposta  Reserva teste independente (não treina)
  /avaliar                     Mede generalização nos testes independentes
  /praticar [n]                Revê até n exemplos antigos (máximo 30)
  /pesquisar assunto           Consulta Wikipédia (opcional, não treina)
  /crescer [n]                 Expansão experimental de canais MLP (8..256)
  /cerebro                     Inspeciona dimensões e progresso
  /guardar                     Guarda checkpoint
  /ajuda                       Mostra comandos
  /sair                        Guarda e termina
'''


def pair(text,sep='=>'):
    if sep not in text: raise ValueError('Utiliza o separador '+sep)
    a,b=text.split(sep,1)
    if not a.strip() or not b.strip():raise ValueError('Falta texto antes ou depois do separador')
    return a.strip(),b.strip()


def handle(brain,line):
    cmd,_,body=line.partition(' ')
    body=body.strip()
    if cmd=='/ajuda':return HELP
    if cmd=='/cerebro':return json.dumps(brain.status(),ensure_ascii=False,indent=2)
    if cmd=='/guardar':brain.core.save();return 'Checkpoint guardado.'
    if cmd=='/ensinar':
        q,a=pair(body)
        loss=brain.teach(q,a)
        return f'Exemplo ensinado. Pesos atualizados. Perda neuronal: {loss:.4f} (não mede compreensão).'
    if cmd=='/corrigir':
        if not brain.last_question: return 'Faz uma pergunta primeiro.'
        loss=brain.teach(brain.last_question,body)
        return f'Correção aprendida nos pesos. Perda: {loss:.4f}.'
    if cmd=='/perguntar' or cmd=='/neuronal':
        if not body:return 'Escreve uma pergunta.'
        ans,source=brain.answer(body,neural_only=cmd=='/neuronal')
        return f'Nord Brain: {ans}\n[origem: {source}]'
    if cmd=='/conceito':
        term,definition=pair(body, '=')
        brain.memory.concept(term,definition)
        return 'Definição guardada como conhecimento simbólico; ainda não treina a rede.'
    if cmd=='/teste':
        q,a=pair(body)
        brain.memory.holdout(q,a)
        return 'Teste independente guardado (não será incluído no treino).'
    if cmd=='/avaliar':
        result=brain.evaluate()
        if result is None:return 'Cria testes primeiro com /teste pergunta => resposta.'
        c,n,s=result
        return f'Testes independentes: {c}/{n} ({s:.0%}). Respostas geradas apenas pela rede neuronal.'
    if cmd=='/praticar':
        n=max(1,min(30,int(body or '10')))
        count,loss=brain.practice(n)
        return f'Revistos {count} exemplos. Perda média: {loss:.4f}.' if count else 'Sem exemplos para praticar.'
    if cmd=='/crescer':
        extra=int(body or '16')
        before=brain.core.parameter_count
        size=brain.grow(extra)
        return f'Expansão experimental: MLP com {size} canais por camada, +{brain.core.parameter_count-before} pesos. Ainda precisam de treino.'
    if cmd=='/pesquisar':
        if os.environ.get('NORD_INTERNET','off').lower()!='on':
            return 'Internet desligada. Ativa NORD_INTERNET=on para permitir pesquisa manual.'
        results=search(body)
        for item in results:brain.memory.observation(body,item['title'],item['url'],item['summary'])
        if not results:return 'Não foram encontrados resultados.'
        return '\n\n'.join(f"{x['title']}\n{x['summary'][:400]}\nFonte: {x['url']}" for x in results) + '\n[Fontes externas não verificadas; não houve treino neuronal]'
    return 'Comando desconhecido. Usa /ajuda.'


def main(argv=None):
    parser=argparse.ArgumentParser(description='Nord Brain: aprendizagem neuronal local e experimental')
    parser.add_argument('--data',type=Path,default=Path(os.environ.get('NORD_DATA',str(Path.home()/'.nord-brain'))))
    args=parser.parse_args(argv)
    brain=Brain(args.data)
    print('Nord Brain v0.2 | Transformer próprio sem pré-treino | /ajuda para comandos')
    print('Aviso: pesos aleatórios; a rede ainda não compreende português e pode produzir texto sem sentido.')
    try:
        while True:
            try:line=input('nord> ').strip()
            except EOFError:break
            if not line:continue
            if line.lower()=='/sair':break
            try:
                if not line.startswith('/'):
                    answer,source=brain.answer(line)
                    print(f'Nord Brain: {answer}\n[origem: {source}]')
                else:print(handle(brain,line))
            except (ValueError,OverflowError) as e:
                print('Erro:',e)
            except (OSError,ConnectionError,TimeoutError) as e:
                print('Falha de rede/ficheiro:',e)
    except KeyboardInterrupt:
        print('\nA terminar...')
    finally:
        brain.close()
        print('Cérebro guardado.')


if __name__=='__main__':main()
