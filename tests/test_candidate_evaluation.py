from nord_brain.candidate_evaluation import choose_candidate, score_candidate

def test_rejects_unintelligible_output():
    assert score_candidate('r e prruam que pano me qudarto', 'sobre qualquer coisa') is None
    assert choose_candidate(['r e prruam que pano me qudarto'], 'sobre qualquer coisa') is None

def test_rejects_empty_candidates():
    assert choose_candidate([], 'Olá') is None
