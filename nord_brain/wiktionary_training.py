"""Convert approved Wiktionary JSONL into Nord linguistic training text.

The input is a local Kaikki/Wiktextract JSONL export. No remote content is
executed, and definitions are not treated as verified personal knowledge.
"""
from __future__ import annotations
import json
from pathlib import Path

def convert(source: str, destination: str, max_entries: int = 250) -> dict:
    if not 1 <= max_entries <= 1000:
        raise ValueError('Limite de entradas inválido')
    path=Path(source).expanduser()
    if not path.is_file() or path.suffix not in ('.jsonl','.json'):
        raise ValueError('É necessário um ficheiro Wikcionário .jsonl')
    target=Path(destination).expanduser()
    target.parent.mkdir(parents=True,exist_ok=True)
    lines=[]
    count=0
    with path.open('r',encoding='utf-8') as handle:
        for raw in handle:
            if count>=max_entries:
                break
            if len(raw)>100000:
                continue
            try:
                entry=json.loads(raw)
            except (ValueError,TypeError):
                continue
            word=str(entry.get('word','')).strip()
            lang=str(entry.get('lang_code','')).lower()
            if lang not in ('pt','por') or not 1<=len(word)<=70:
                continue
            senses=entry.get('senses',[])
            for sense in senses[:3]:
                definitions=sense.get('glosses') or sense.get('raw_glosses') or []
                for definition in definitions[:2]:
                    definition=' '.join(str(definition).split())
                    if not 8<=len(definition)<=320:
                        continue
                    lines.append('Palavra: '+word+'. Significado: '+definition+'.')
                    count+=1
                    break
                if count>=max_entries:
                    break
    if not lines:
        raise ValueError('Não foram encontradas definições em português')
    target.write_text('\n'.join(lines),encoding='utf-8')
    return {'entries':count,'path':str(target)}
