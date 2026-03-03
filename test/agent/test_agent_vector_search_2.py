"""E2E checks for vector_search lecture schedule cases through agent()."""

import pytest

from src.agent import agent

pytestmark = [pytest.mark.agent, pytest.mark.skip]


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('когда проходят лекции по мл', 'по пятницам, 11:10 - 12:30'),
        ('время лекций по машинному обучению', 'по пятницам, 11:10 - 12:30'),
        ('лекции по машинке проходят время', 'по пятницам, 11:10 - 12:30'),
        ('расписание лекций по мл', 'по пятницам, 11:10 - 12:30'),
        ('лекционная часть по мл время', 'по пятницам, 11:10 - 12:30'),
        ('в какое время проходят лекции по машинке', 'по пятницам, 11:10 - 12:30'),
        ('в какой день и время лекции по мл', 'по пятницам, 11:10 - 12:30'),
    ],
)
def test_agent_e2e_vector_search_ml_time(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        (
            'когда проходят лекции по теории вероятности',
            'Пятница     | 10:30 – 12:00',
        ),
        (
            'время лекций по теорверу',
            'Пятница     | 10:30 – 12:00',
        ),
        (
            'день и время лекций по теорверу',
            'Пятница     | 10:30 – 12:00',
        ),
        (
            'расписание лекций по вероятности',
            'Пятница     | 10:30 – 12:00',
        ),
        (
            'лекционная часть по вероятности время',
            'Пятница     | 10:30 – 12:00',
        ),
    ],
)
def test_agent_e2e_vector_search_prob_time(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('когда проходят лекции по оптимизации', 'вторник, лекция в 13:00'),
        ('время лекций по оптам', 'вторник, лекция в 13:00'),
        ('оптимизация день и время лекций', 'вторник, лекция в 13:00'),
        ('расписание лекций по оптимизации', 'вторник, лекция в 13:00'),
        ('по метоптам когда идут лекции', 'вторник, лекция в 13:00'),
    ],
)
def test_agent_e2e_vector_search_opt_time(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer
