"""Contextual training must not leak reserved examples."""
from nord_brain.contextual_dataset import examples

def test_contextual_examples_are_distinct():
    train = examples(512)
    heldout = examples(512, holdout=True)
    assert len(train) >= 10
    assert len(heldout) >= 4
    assert not ({q for q, _ in train} & {q for q, _ in heldout})

def test_same_phrase_can_depend_on_context():
    train = examples(512)
    relevant = [(q,a) for q,a in train if 'qualquer' in q.casefold()]
    assert len(relevant) >= 2
    assert len({a for _,a in relevant}) >= 2

def test_prompt_respects_small_context():
    for context in (128, 256, 512):
        for prompt, _ in examples(context):
            assert len(prompt.encode('utf-8')) <= max(24,context-78)
