"""T2.14-T2.19: real MiniLM + real Ollama. Skipped with a reason when Ollama is not running."""

import pytest

pytestmark = pytest.mark.integration


def cited_pages(answer):
    return {p for c in answer.citations for p in c.pages}


def explain(answer):
    return f"{answer.answer!r} conf={answer.confidence} pages={sorted(cited_pages(answer))}"


def test_t2_14_refund_window(ask):
    a = ask("What's the refund window?")
    assert not a.refused, explain(a)
    assert "30" in a.answer, explain(a)
    assert 12 in cited_pages(a) and "[p.12]" in a.answer, explain(a)
    assert a.confidence >= 0.5, explain(a)


def test_t2_15_employee_discount(ask):
    a = ask("What is the employee discount?")
    assert not a.refused and "15%" in a.answer, explain(a)
    assert 18 in cited_pages(a), explain(a)


def test_t2_16_warranty_claim_spans_pages(ask):
    a = ask("How do I file a warranty claim?")
    assert not a.refused, explain(a)
    assert cited_pages(a) & {20, 21}, explain(a)


def test_t2_17_off_topic_is_refused(ask):
    a = ask("Who won the 2018 FIFA World Cup?")
    assert a.refused, explain(a)


def test_t2_18_absent_fact_is_refused(ask):
    a = ask("What is the CEO's salary?")
    assert a.refused, explain(a)


def test_t2_19_same_question_same_answer(ask):
    first = ask("What are the store hours?")
    second = ask("What are the store hours?")
    assert first.answer == second.answer
