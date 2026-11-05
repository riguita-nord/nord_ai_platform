import json
from nord_brain.wiktionary_training import convert

def test_converts_portuguese_definitions(tmp_path):
    source=tmp_path/'words.jsonl'
    source.write_text(
        json.dumps({'word':'banco','lang_code':'pt',
                    'senses':[{'glosses':['Assento onde as pessoas se podem sentar.']},
                              {'glosses':['Instituição que presta serviços financeiros.']}]})
        +'\n'+json.dumps({'word':'house','lang_code':'en',
                            'senses':[{'glosses':['A building to live in.']}]})+'\n',
        encoding='utf-8')
    destination=tmp_path/'corpus.txt'
    result=convert(str(source),str(destination))
    text=destination.read_text(encoding='utf-8')
    assert result['entries']==2
    assert 'Assento' in text
    assert 'Instituição' in text
    assert 'house' not in text
