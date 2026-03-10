"""E2E checks for get_top_students answers through agent()."""

import pytest

from src.agent import agent
from src.api.get_top_students import get_top_students

pytestmark = [pytest.mark.agent]


def _format_top(subject_name: str, k: int) -> str:
    top_students = get_top_students(subject_name, k)
    return ', '.join(
        f'{student["name"]} ({student["score"]:.1f})' for student in top_students
    )


def ml_top(k: int = 3) -> str:
    return _format_top('Machine Learning', k)


def prob_top(k: int = 3) -> str:
    return _format_top('Probability Theory', k)


def opt_top(k: int = 3) -> str:
    return _format_top('Optimization Theory', k)


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('машинное обучение: лучшие студенты', ml_top()),
        ('в машинном обучении покажи лучших студентов', ml_top()),
        ('по машинному обучению топ-4 студентов', ml_top(4)),
        ('теория вероятностей: кто лучшие студенты? топ3', prob_top(3)),
        ('лучшие в теории вероятностей студенты', prob_top()),
        ('топ 10 по теории вероятностей среди студентов', prob_top(10)),
        ('теория оптимизации: кто из студентов лучший?', opt_top(1)),
        ('среди студентов лучшие по теории оптимизации', opt_top()),
        ('по теории оптимизации покажи студенческий топ-10', opt_top(10)),
    ],
)
def test_agent_e2e_get_top_students_simple(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('топовые студенты по ml', ml_top()),
        ('лучшие топ3 студенты по мл?', ml_top(3)),
        ('топ2 студентов по машинке', ml_top(2)),
        ('лучшие по машинному обучению топ7', ml_top(7)),
        ('три лучших студента по оптимизации', opt_top(3)),
        ('топ-3 лучших студентов по оптам', opt_top(3)),
        ('топовые студенты по метоптам?', opt_top()),
        ('топ учащихся по мл', ml_top()),
        ('топ студентов по методам оптимизации', opt_top()),
        ('топ4 лучших студентиков по теорверу', prob_top(4)),
        ('лучшие топовые студенты по терверу?', prob_top()),
        ('top студентов по теории вероятности', prob_top()),
        ('топ пять учеников по мл', ml_top(5)),
        ('топовые восемь студентов, предмет тервер', prob_top(8)),
        ('по предмету мл топ два учащихся', ml_top(2)),
    ],
)
def test_agent_e2e_get_top_students_complex(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer
