"""Regression tests for informal European Portuguese social exchanges."""
from nord_brain.social_dialogue import social_act, social_response

def test_reciprocal_question_after_greeting():
    assert social_act('tou bem e tu?', 'Olá! Como estás hoje?') == 'reciprocal'
    assert social_act('tou bem e tu', 'Olá! Como estás hoje?') == 'reciprocal'
    assert social_act('tou bem pois e tu?', 'Olá! Como estás hoje?') == 'reciprocal'

def test_answer_vs_question_by_context():
    assert social_act('tudo bem', 'Olá! Como estás hoje?') == 'answer'
    assert social_act('tudo bem', 'Olá!') == 'question'
    assert social_act('tou bem', 'Olá! Como estás hoje?') == 'answer'

def test_unrelated_topics_abstain():
    assert social_act('como configurar o servidor?', 'Olá!') is None
    assert social_act('preciso de ajuda com Python', 'Olá!') is None

def test_social_reply_never_requests_fact_training():
    for act in ('reciprocal','answer','question','acknowledge'):
        reply=social_response(act)
        assert reply
        assert 'pesquise fontes' not in reply
        assert 'Podes ensinar-me' not in reply
