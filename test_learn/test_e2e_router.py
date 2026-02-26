"""E2E tests for router behavior through agent()."""

import pytest

from src_learn.agent import agent

pytestmark = [pytest.mark.llm, pytest.mark.e2e]


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        (
            'машинное обучение: лучшие студенты',
            'Avery (80.4), John (76.9), Ethan (76.5)',
        ),
        (
            'в машинном обучении покажи лучших студентов',
            'Avery (80.4), John (76.9), Ethan (76.5)',
        ),
        (
            'по машинному обучению топ-4 студентов',
            'Avery (80.4), John (76.9), Ethan (76.5), Emma (75.8)',
        ),
        (
            'теория вероятностей: кто лучшие студенты? топ3',
            'Maria (79.1), Mia (74.9), Abigail (74.6)',
        ),
        (
            'лучшие в теории вероятностей студенты',
            'Maria (79.1), Mia (74.9), Abigail (74.6)',
        ),
        (
            'топ 10 по теории вероятностей среди студентов',
            'Maria (79.1), Mia (74.9), Abigail (74.6), Ella (73.8), David (73.4), Mila (72.6), Emily (72.4), Liam (71.4), Sofia (70.6), Ava (70.5)',
        ),
        (
            'теория оптимизации: кто из студентов лучший?',
            'Harper (78.6), Evelyn (77.6), Mia (76.8)',
        ),
        (
            'среди студентов лучшие по теории оптимизации',
            'Harper (78.6), Evelyn (77.6), Mia (76.8)',
        ),
        (
            'по теории оптимизации покажи студенческий топ-10',
            'Harper (78.6), Evelyn (77.6), Mia (76.8), Alice (76.3), Avery (76.2), Charlotte (75.8), Bob (75.3), Ava (75.2), Elizabeth (74.6), Victoria (74.5)',
        ),
    ],
)
def test_agent_e2e_database_tool_top_students_simple(query: str, expected_answer: str):
    result = agent([{'role': 'user', 'content': query}])
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        (
            'топовые студенты по ml',
            'Avery (80.4), John (76.9), Ethan (76.5)',
        ),
        (
            'лучшие топ3 студенты по мл?',
            'Avery (80.4), John (76.9), Ethan (76.5)',
        ),
        (
            'топ2 студентов по машинке',
            'Avery (80.4), John (76.9)',
        ),
        (
            'лучшие по машинному обучению топ7',
            'Avery (80.4), John (76.9), Ethan (76.5), Emma (75.8), Liam (75.5), Bob (74.9), Sofia (74.8)',
        ),
        (
            'три лучших студента по оптимизации',
            'Harper (78.6), Evelyn (77.6), Mia (76.8)',
        ),
        (
            'топ-3 лучших студентов по оптам',
            'Harper (78.6), Evelyn (77.6), Mia (76.8)',
        ),
        (
            'топовые студенты по метоптам?',
            'Harper (78.6), Evelyn (77.6), Mia (76.8)',
        ),
        (
            'топ учащихся по мл',
            'Avery (80.4), John (76.9), Ethan (76.5)',
        ),
        (
            'топ студентов по методам оптимизации',
            'Harper (78.6), Evelyn (77.6), Mia (76.8)',
        ),
        (
            'топ4 лучших студентиков по теорверу',
            'Maria (79.1), Mia (74.9), Abigail (74.6), Ella (73.8)',
        ),
        (
            'лучшие топовые студенты по терверу?',
            'Maria (79.1), Mia (74.9), Abigail (74.6)',
        ),
        (
            'top студентов по теории вероятности',
            'Maria (79.1), Mia (74.9), Abigail (74.6)',
        ),
        (
            'топ пять учеников по мл',
            'Avery (80.4), John (76.9), Ethan (76.5), Emma (75.8), Liam (75.5)',
        ),
        (
            'топовые восемь студентов, предмет тервер',
            'Maria (79.1), Mia (74.9), Abigail (74.6), Ella (73.8), David (73.4), Mila (72.6), Emily (72.4), Liam (71.4)',
        ),
        (
            'по предмету мл топ два учащихся',
            'Avery (80.4), John (76.9)',
        ),
    ],
)
def test_agent_e2e_database_tool_top_students_complex(query: str, expected_answer: str):
    result = agent([{'role': 'user', 'content': query}])
    assert str(result.get('answer') or '') == expected_answer
