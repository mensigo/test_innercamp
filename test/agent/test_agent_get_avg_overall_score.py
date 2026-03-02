"""E2E checks for get_avg_overall_score answers through agent()."""

import pytest

from src.agent import agent

pytestmark = [pytest.mark.llm, pytest.mark.e2e]


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('средний балл по всем предметам', '4.0'),
        ('по предметам среднее', '4.0'),
        ('среднее по курсам', '4.0'),
        ('усредненный балл по всем курсам', '4.0'),
        ('все предметы, средний балл', '4.0'),
    ],
)
def test_agent_e2e_get_avg_overall_score_simple(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('средний скор по предметам', '4.0'),
        ('скор усредненный по курсам', '4.0'),
        ('по дисциплинам всем среднее', '4.0'),
    ],
)
def test_agent_e2e_get_avg_overall_score_complex(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer
