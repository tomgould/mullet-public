
from mullet.knowledge.engine.search.contextualize import query_contextualize


def test_empty_history():
    assert query_contextualize("hello", []) == "hello"


def test_empty_query():
    assert query_contextualize("", ["prior"]) == ""


def test_self_contained():
    assert query_contextualize("Tell me about the Eiffel Tower", ["What is the capital of France?"]) == "Tell me about the Eiffel Tower"


def test_follow_up_expanded():
    assert query_contextualize("what about the tests?", ["What is Python?"]) == "What is Python? — what about the tests?"


def test_short_question_follow_up():
    assert query_contextualize("tests?", ["Explain decorators in Python"]) == "Explain decorators in Python — tests?"


def test_pronoun_follow_up():
    assert query_contextualize("it uses event loops", ["How does asyncio work?", "ok"]) == "How does asyncio work? — it uses event loops"


def test_no_substantive_prior():
    assert query_contextualize("what about that?", ["ok", "sure", "and?"]) == "what about that?"


def test_prior_follow_up():
    assert query_contextualize("how about decorators?", ["What is Python?", "what about the tests?"]) == "What is Python? — how about decorators?"
