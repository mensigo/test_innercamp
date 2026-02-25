"""Intent classification for src_learn agent."""

from src_learn.config import config
from src_learn.logger import logger
from src_learn.utils import post_chat_completions


def classify_intent(user_query: str) -> bool:
    """Classify query relevance for student-grade-exam functionality."""
    if not user_query:
        return False

    system_prompt = """
Ты классифицируешь пользовательский запрос для локального агента учебной аналитики.

Верни СТРОГО одно слово:
- relevant
- irrelevant

Считай запрос relevant, если он относится ХОТЯ БЫ к одному из пунктов:
- студенты, предметы, оценки, итоговые баллы, топ студентов;
- аналитика по экзаменам: самые сложные вопросы, статистика провалов, failed/meta-оценки;
- теоретические вопросы по предметам агента: Machine Learning, Probability Theory, Optimization Theory.

ВАЖНО: теоретические формулировки тоже relevant, даже если в тексте нет слов "студент" или "оценка".
Примеры relevant:
- "Назовите 4 ключевые особенности машинного обучения"
- "Сформулируйте теорему Байеса"
- "Что такое условия ККТ в оптимизации?"
- "Что такое переобучение?"
- "Что такое случайная величина?"

Для всех остальных запросов возвращай irrelevant.
"""

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt.strip()},
            {'role': 'user', 'content': user_query},
        ],
        'temperature': config.freezing,
        'max_tokens': 5,
    }
    response = post_chat_completions(payload, verbose=config.debug)
    if 'error' in response:
        logger.warning(f'agent // classify error: {response["error"]}')
        return False

    try:
        content = (
            response.get('choices', [{}])[0]
            .get('message', {})
            .get('content', '')
            .strip()
            .lower()
        )
    except (AttributeError, IndexError):
        return False

    return 'relevant' in content and 'irrelevant' not in content
