"""Terminal conversation first. Diagnostic commands are optional."""
from __future__ import annotations
import argparse
import json
import os
import shlex
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
  Cria um livro sobre uma cidade no espaço. (escreve por capítulos)
  Conta-me uma história longa com 12 capítulos sobre robôs.
  Continua a história. / Mostra capítulo 2. / Lista livros.
  Quem é Elon Musk? / Pesquisa sobre um tema. (fontes públicas; não são verdades confirmadas)\n  Confirma fonte 1. (aprendizagem neuronal após a tua validação)
  2+2 é 4. (gera hipóteses matemáticas e treina autonomamente)
  Simula 3 cenários.  (testa 4 ações e aprende com as consequências)
  Experimenta tudo.  (matemática, padrões, lógica, linguagem e mundo virtual)
  Experimenta lógica 2.  (2 hipóteses de lógica ensinada)
  Mostra teorias.  (ver hipóteses, transformações e estados)
  Quero aprovar as hipóteses.  (treino matemático com aprovação)
  Aprova hipótese 3+2. / Rejeita hipótese 3+2.
  Pausa aprendizagem. / Retoma aprendizagem.

Comandos técnicos opcionais: /cerebro, /avaliar, /ajuda, /sair
  /treinar-portugues 20    (treino REAL nos pesos, CPU, com avaliação separada)
  /treinar-texto <ficheiro.txt> [6]  (treino de texto corrido que TU aprovas)
  /gerar-neural <tema>    (ver saída direta do teu Transformer, sem verificadores)
  /avaliar-portugues      (perda em frases não usadas no currículo)
  /treinar-dialogos 12    (treino contextual nos pesos do Transformer, 1–48 passos)
  /avaliar-dialogos       (perda em conversas portuguesas reservadas)
  /diagnostico-intencao <frase> (previsão da cabeça neuronal de intenção)
  /diagnostico-portugues   (compara as duas avaliações de português)
  /diagnostico-pesquisa   (testar pesquisa na Internet e fornecedores)
  /livro novo <tema>        (começa um romance de 10 capítulos)
  /livro novo 12 <tema>     (escolhe o número de capítulos, 3–24)
  /livro continuar [1-3]  (escreve e guarda novos capítulos)
  /livro estado           (plano e progresso)
  /livro listar           (livros guardados)
  /livro selecionar <id>  (retomar um livro anterior)
  /livro ler <n>          (reler um capítulo)
  /livro exportar         (exporta Markdown para a pasta local)
  /diagnostico-linguagem  (mostrar o Transformer Nord e os passos de treino)
  /contexto             (ver o último assunto guardado)
  /limpar-contexto       (apagar apenas o histórico de conversa)
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
    if line.startswith('/simular'):
        parts=line.split()
        if len(parts)>2:raise ValueError('Usa /simular ou /simular 5')
        amount=int(parts[1]) if len(parts)==2 else 3
        if not 1<=amount<=10:raise ValueError('Entre 1 e 10 cenários')
        return brain.simulate_world(amount)[0]
    if line.startswith('/laboratorio'):
        parts=line.split()
        if len(parts)>3:raise ValueError('Usa /laboratorio tudo 1')
        domain=parts[1] if len(parts)>1 else 'tudo'
        amount=int(parts[2]) if len(parts)>2 else 1
        return brain.simulate_cognition(domain,amount)[0]
    if line.startswith('/treinar-portugues'):
        from .portuguese_training import train, describe
        parts=line.split()
        if len(parts)>2:raise ValueError('Usa /treinar-portugues ou /treinar-portugues 20')
        amount=int(parts[1]) if len(parts)==2 else 20
        return describe(train(brain,amount))
    if line.startswith('/treinar-texto'):
        from .native_training import train_approved_file, describe
        args=shlex.split(line)
        if len(args) not in (2,3):
            raise ValueError('Usa /treinar-texto caminho/ficheiro.txt [1-32]')
        updates=int(args[2]) if len(args)==3 else 6
        return describe(train_approved_file(brain,args[1],updates=updates))
    if line.startswith('/gerar-neural'):
        prompt=line.removeprefix('/gerar-neural').strip()
        if not prompt:
            raise ValueError('Usa /gerar-neural <pergunta ou tema>')
        with brain.lock:
            output=brain.core.generate(prompt,max_bytes=125,temperature=.7)
        return ('Saída BRUTA dos pesos Nord (não verificada): '
                +(output or '[sem saída; ainda não sabe gerar texto útil]'))
    if line=='/avaliar-portugues':
        from .portuguese_training import evaluate
        result=evaluate(brain)
        return ('Ainda não existem perguntas de avaliação independentes.' if result is None
                else f"Perda em português: {result['loss']} ({result['examples']} perguntas nunca treinadas). "
                     'Isto não equivale a fluência; avalia apenas a previsão dos tokens.')
    if line.startswith('/treinar-dialogos'):
        from .dialogue_training import train,describe
        parts=line.split()
        if len(parts)>2:raise ValueError('Usa /treinar-dialogos ou /treinar-dialogos 12')
        updates=int(parts[1]) if len(parts)==2 else 12
        return describe(train(brain,updates))
    if line in ('/avaliar-dialogos','/diagnostico-portugues'):
        from .dialogue_training import evaluate
        from .portuguese_training import evaluate as old_eval
        data=evaluate(brain)
        old=old_eval(brain)
        warning=(' A perda de português é elevada: os pesos atuais podem '
                 'ter sofrido interferência entre os diferentes treinos.'
                 if old and old['loss']>7 else '')
        return (f"Diálogos com contexto: perda {data['dialogue_loss']} "
                f"em {data['heldout_dialogues']} exemplos reservados. "
                f"Conversa natural: perda {data['natural_loss']} "
                f"em {data['natural_holdouts']} exemplos reservados. "
                f"Português geral: perda {old['loss'] if old else 'n/d'} "
                f"em {old['examples'] if old else 0} perguntas. "
                'Apenas métricas de previsão, não fluência.'
                +warning)
    if line.startswith('/diagnostico-intencao'):
        text=line.removeprefix('/diagnostico-intencao').strip()
        if not text:raise ValueError('Usa /diagnostico-intencao <frase>')
        last=brain.dialogue.latest()
        focus=last['focus'] if last else 'dialogue'
        kind,confidence=brain.intent.predict(text,focus)
        return (f'Assunto: {focus}. Intenção neuronal: '
                f'{kind or "incerta / nova questão"}; '
                f'confiança={confidence:.2f}. '
                f'Acurácia em frases reservadas={brain.intent.validation_accuracy:.2f}.')
    if line.startswith('/diagnostico-pesquisa'):
        from .internet import search_diagnostic
        term=line.removeprefix('/diagnostico-pesquisa').strip() or 'Elon Musk'
        return search_diagnostic(term)
    if line=='/diagnostico-linguagem':
        return (brain.language.status()+'. Passos de treino nos pesos Nord: '
                +str(brain.core.steps)+'. Pode usar /gerar-neural para testar geração direta.')
    if line.startswith('/livro'):
        from .story_studio import BookCommand
        args=line[len('/livro'):].strip().split()
        if not args or args[0] in ('ajuda','help'):
            return ('Usa /livro novo [3-24] <tema>, /livro continuar [1-3], '
                    '/livro estado, /livro listar, /livro selecionar <id>, '
                    '/livro ler <n> ou /livro exportar.')
        op=args[0].casefold()
        if op in ('novo','criar'):
            chapters=int(args[1]) if len(args)>1 and args[1].isdigit() else 10
            start=2 if len(args)>1 and args[1].isdigit() else 1
            theme=' '.join(args[start:]).strip()
            return brain.book_operation(BookCommand('start',theme,chapters))[0]
        if op in ('continuar','continua'):
            return brain.book_operation(BookCommand('continue',amount=int(args[1]) if len(args)>1 else 1))[0]
        if op in ('estado','plano'):
            return brain.book_operation(BookCommand('status'))[0]
        if op in ('listar','lista'):
            return brain.book_operation(BookCommand('list'))[0]
        if op in ('selecionar','abrir') and len(args)==2:
            return brain.book_operation(BookCommand('select',amount=int(args[1])))[0]
        if op in ('ler','mostrar') and len(args)==2:
            return brain.book_operation(BookCommand('read',amount=int(args[1])))[0]
        if op in ('exportar','guardar'):
            return brain.book_operation(BookCommand('export'))[0]
        raise ValueError('Comando de livro inválido. Usa /livro ajuda.')
    if line=='/contexto':
        latest=brain.dialogue.latest()
        if latest is None:return 'Ainda não existe contexto de conversa guardado.'
        return (f'Último assunto ({latest["focus"]}): {latest.get("topic") or latest["question"]}\n'
                'A memória de conversa é local e separada das lições neuronais. '
                'Usa /limpar-contexto para apagá-la.')
    if line=='/limpar-contexto':
        brain.dialogue.clear()
        brain.language.history.clear()
        return 'Contexto recente apagado. Os livros e as lições aprendidas foram preservados.'
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
    print('Nord Brain v2.1 | intenções neuronais + treino de diálogos PT + memória contextual')
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
