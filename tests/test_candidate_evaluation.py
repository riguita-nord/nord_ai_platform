from nord_brain.candidate_evaluation import choose_candidate, score_candidate

def test_rejects_unintelligible_output():
    assert score_candidate('r e prruam que pano me qudarto', 'sobre qualquer coisa') is None
    assert choose_candidate(['r e prruam que pano me qudarto'], 'sobre qualquer coisa') is None

def test_rejects_empty_candidates():
    assert choose_candidate([], 'Olá') is None

def test_rejects_lone_function_word():
    assert score_candidate('e', 'tree?') is None
    assert score_candidate('a', 'tree?') is None
    assert choose_candidate(['e'], 'tree?') is None
    assert choose_candidate(['e', 'Sim'], 'tree?') == 'Sim'

def test_accepts_coherent_portuguese_with_many_short_words():
    answer = 'A \u00e1gua \u00e9 essencial \u00e0 vida e deve ser mantida limpa.'
    question = 'Para que precisamos de beber \u00e1gua?'
    assert score_candidate(answer, question) is not None
    assert choose_candidate([answer], question) == answer

def test_rejects_pseudo_portuguese_despite_function_words():
    question = 'O que \u00e9 uma \u00e1rvore?'
    samples = [
        'eutaaosbs : a etaamvesba e ePaca aebP m',
        'rtea e de ca aacea drare e u s ca aa co',
        'ereCotes tam tage etema utunes te ad l a n',
        'ardetra lae e deriante te etr e a o aco',
    ]
    for sample in samples:
        assert score_candidate(sample, question) is None
