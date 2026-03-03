"""E2E checks for get_avg_score answers through agent()."""

import pytest

from src.agent import agent

pytestmark = [pytest.mark.agent]


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('средний балл по машинному обучению', '4.1'),
        ('средний скор по теории вероятности', '3.9'),
        ('усредненный скор по оптимизации', '4.0'),
        ('по теории вероятности средняя оценка', '3.9'),
        ('покажи среднее по методам оптимизации', '4.0'),
    ],
)
def test_agent_e2e_get_avg_score_simple(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('среднее по мл', '4.1'),
        ('средний балл, опты', '4.0'),
        ('тервер среднее', '3.9'),
        ('машинка скор с усреднением', '4.1'),
        ('выведи балл как среднее по метоптам', '4.0'),
    ],
)
def test_agent_e2e_get_avg_score_complex(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer
