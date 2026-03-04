"""E2E checks for vector_search lecture location cases through agent()."""

import pytest

from src.agent import agent

pytestmark = [pytest.mark.agent]


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('где проходят лекции по мл', 'П8а'),
        ('аудитория лекций по машинному обучению', 'П8а'),
        ('лекции по машинке место', 'П8а'),
        ('расписание лекций по мл, какая аудитория', 'П8а'),
    ],
)
def test_agent_e2e_vector_search_ml_place(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('где проходят лекции по теории вероятности', 'R302'),
        ('место лекций по теорверу', 'R302'),
        ('по расписанию лекции по теорверу проходят в ?', 'R302'),
        ('по вероятности лекции где', 'R302'),
    ],
)
def test_agent_e2e_vector_search_prob_place(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('где проходят лекции по оптимизации', 'П9'),
        ('ауд лекций по оптам', 'П9'),
        ('оптимизация аудитория лекций', 'П9'),
        ('лекции по оптимизации проходят в аудитории номер ?', 'П9'),
    ],
)
def test_agent_e2e_vector_search_opt_place(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer
