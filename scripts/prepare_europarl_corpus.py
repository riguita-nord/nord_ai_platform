"""Extract UTF-8 sentence text from the Språkbanken Europarl-PT XML archive."""
from __future__ import annotations

import argparse
import bz2
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def sentences(source: Path, limit: int | None = None):
    root = None
    with bz2.open(source, 'rb') as stream:
        for index, sentence in enumerate(_parse(stream), start=1):
            yield sentence
            if limit is not None and index >= limit:
                break


def _parse(stream):
    root = None
    for event, elem in ET.iterparse(stream, events=('start', 'end')):
        if event == 'start' and root is None:
            root = elem
            continue
        if event != 'end':
            continue
        if elem.tag == 'sentence':
            value = ' '.join(part.strip() for part in elem.itertext() if part.strip())
            value = re.sub(r'\s+([,.;:!?%])', r'\1', value)
            value = re.sub(r'([«([{])\s+', r'\1', value)
            value = re.sub(r'\s+', ' ', value).strip()
            if value:
                yield value
            elem.clear()
        elif elem.tag == 'text':
            elem.clear()
            if root is not None:
                root.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/corpus/europarl-pt.xml.bz2'))
    parser.add_argument('--output', type=Path, default=Path('data/corpus/europarl-pt.txt'))
    parser.add_argument('--max-sentences', type=int, default=500_000)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Arquivo não encontrado: {args.input}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.output.open('w', encoding='utf-8', newline='\n') as stream:
        for sentence in sentences(args.input, args.max_sentences or None):
            stream.write(sentence + '\n')
            total += 1
            if total % 100_000 == 0:
                stream.flush()
                print(f'{total:,} frases', flush=True)
    print(f'Concluído: {total:,} frases; {args.output.stat().st_size:,} bytes UTF-8')


if __name__ == '__main__':
    main()
