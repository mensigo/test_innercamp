"""E2E checks for search_rag answers through agent()."""

import pytest

from src.agent import agent

pytestmark = [pytest.mark.llm, pytest.mark.agent]


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('лектор по мл', ''),
        ('лектор по теории вероятностей', ''),
        ('лектор по методам оптимизации', ''),
        ('лектор по машинному обучению', ''),
        ('лектор по оптимизации', ''),
        ('лектор по теории вероятностей', ''),
        ('лектор по методам оптимизации', ''),
        ('лектор по машинному обучению', ''),
        ('лектор по оптимизации', ''),
    ],
)
def test_agent_e2e_search_rag_simple(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('лектор по мл', ''),
        ('лектор по теории вероятностей', ''),
        ('лектор по методам оптимизации', ''),
        ('лектор по машинному обучению', ''),
        ('лектор по оптимизации', ''),
        ('лектор по теории вероятностей', ''),
        ('лектор по методам оптимизации', ''),
        ('лектор по машинному обучению', ''),
        ('лектор по оптимизации', ''),
    ],
)
def test_agent_e2e_search_rag_complex(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer
