"""Tests for route_query in src_learn.router."""

import pytest

from src_learn.router import route_query

pytestmark = [pytest.mark.llm]


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
def test_route_query_top_students(query: str):
    result = route_query(query)
    assert result.get('tool_name') == 'database_tool'
    assert result.get('operation') == 'top_students'


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
def test_route_query_top_students_complex_phrases(query: str):
    result = route_query(query)
    assert result.get('tool_name') == 'database_tool'
    assert result.get('operation') == 'top_students'
