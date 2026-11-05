"""Shared serialization for dialogue training and inference.

Keep the newest user request and preceding turns inside the byte-level model
window. This does not increase model capacity or guarantee fluent language.
"""
from __future__ import annotations

def prompt_from_history(message: str, history: list[dict], context: int) -> str:
    budget=max(24,context-78)
    turns=[]
    for row in history[-8:]:
        role=row.get('role')
        if role in ('user','assistant'):
            turns.append(('Tu' if role=='user' else 'Nord')+': '+str(row.get('content','')).strip())
    turns.append('Tu: '+message.strip())
    selected=[]
    remaining=budget
    for turn in reversed(turns):
        raw=turn.encode('utf-8')
        if len(raw)>remaining:
            if not selected:
                raw=raw[-remaining:]
                selected.insert(0,raw.decode('utf-8','ignore'))
            break
        selected.insert(0,turn)
        remaining-=len(raw)+1
        if remaining<=0:break
    return '\n'.join(selected)

def training_prompt(turns: list[tuple[str,str]], context: int) -> str:
    if len(turns)<2 or turns[-1][0]!='user':
        raise ValueError('O último turno deve pertencer ao utilizador')
    history=[{'role':role,'content':content} for role,content in turns[:-1]]
    return prompt_from_history(turns[-1][1],history,context)
