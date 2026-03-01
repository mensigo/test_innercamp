"""Tests for classify_intent."""

import pytest

from src_example.classify_intent import classify_intent

pytestmark = [pytest.mark.llm]


@pytest.mark.parametrize(
    'query',
    [
        'Расскажи анекдот про кота',
        'Какая погода завтра в Москве?',
        'Напиши письмо работодателю на английском',
        'Сколько будет 123 * 456?',
        'Придумай название для кофейни',
        'Сделай краткое резюме романа "Война и мир"',
        'Напиши SQL-запрос для выборки пользователей за месяц',
        'Объясни, как работает asyncio в Python',
        'Составь план поездки в Стамбул на 3 дня',
        'Какие акции технологических компаний сейчас интересны?',
        'Подбери рецепт пасты с грибами и сливками',
        'Придумай 10 названий для pet-проекта',
        'Как улучшить качество сна без лекарств?',
        'Переведи текст на испанский язык',
        'Напиши мотивационное письмо в магистратуру',
    ],
)
def test_classify_intent_irrelevant(query: str):
    """Irrelevant requests must be classified as not relevant."""
    assert classify_intent(query) is False


@pytest.mark.parametrize(
    'query',
    [
        'Покажи топ-5 студентов по Machine Learning',
        'Какие 3 самых сложных вопроса в Probability Theory?',
        'Сколько вопросов завалены более чем 80% студентов в Optimization Theory?',
        'Покажи meta-оценки студента John по Machine Learning',
        'Какие вопросы студент Maria не сдала по Probability Theory?',
        'Назовите 4 ключевые особенности машинного обучения',
        'Сформулируйте теорему Байеса',
        'Что такое условия ККТ в оптимизации?',
        'Для студента John в Machine Learning найди 3 лучших студентов, сдавших вопросы, которые он завалил',
        'Какая итоговая оценка у Alice по Machine Learning?',
    ],
)
def test_classify_intent_relevant(query: str):
    """Relevant student-domain requests must be classified as relevant."""
    assert classify_intent(query) is True
