"""E2E tests for router behavior through agent()."""

import json
import time

import pytest

from src_learn.agent import IRRELEVANT_MESSAGE, agent

pytestmark = [pytest.mark.llm, pytest.mark.e2e]


def _assert_top_students_answer(result: dict):
    """Validate that agent response matches top-students payload."""
    assert result.get('answer') != IRRELEVANT_MESSAGE
    answer = str(result.get('answer') or '')
    if answer.startswith('Unknown subject:'):
        return
    payload = json.loads(answer or '[]')
    assert isinstance(payload, list)
    assert payload
    first_row = payload[0]
    assert 'name' in first_row
    assert 'grade' in first_row


@pytest.mark.parametrize(
    'query',
    [
        'Машинное обучение: лучшие студенты',
        'В Машинном обучении покажи лучших студентов',
        'По Машинному обучению топ студентов',
        'Теория вероятностей: кто лучшие студенты?',
        'Лучшие в Теории вероятностей студенты',
        'Топ по Теории вероятностей среди студентов',
        'Теория оптимизации: кто из студентов лучший?',
        'Среди студентов лучшие по Теории оптимизации',
        'По Теории оптимизации покажи студенческий топ',
    ],
)
def test_agent_e2e_database_tool_top_students_simple(query: str):
    time.sleep(0.2)
    result = agent([{'role': 'user', 'content': query}])
    _assert_top_students_answer(result)


@pytest.mark.parametrize(
    'query',
    [
        'лучшие студенты по ml',
        'лучшие студенты по мл?',
        'топ студентов по машинке',
        'лучшие по машинному обучению',
        'лучших студентов по оптам',
        'топовые студенты по метоптам?',
        'топ студентов по методам оптимизации',
        'лучших студентиков по теорверу',
        'лучшие студенты по терверу?',
        'топ студентов по теории вероятности',
    ],
)
def test_agent_e2e_database_tool_top_students_complex(query: str):
    time.sleep(0.2)
    result = agent([{'role': 'user', 'content': query}])
    _assert_top_students_answer(result)
