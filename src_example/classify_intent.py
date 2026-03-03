"""Intent classification for example agent."""

from .config import config
from .logger import logger
from .utils import post_chat_completions


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
- запросы про общий средний балл по всем предметам/курсам/дисциплинам.
- запросы про лектора/преподавателя лекций по этим предметам.

ВАЖНО: теоретические формулировки тоже relevant, даже если в тексте нет слов "студент" или "оценка".
ВАЖНО: разговорные алиасы и краткие формы предметов тоже relevant.
Считай relevant формулировки с алиасами:
- "мл", "машинка" -> Machine Learning
- "теорвер", "тервер", "теория вероятности" -> Probability Theory
- "опты", "метопты", "методы оптимизации" -> Optimization Theory
ВАЖНО: запросы про среднее/усреднение/средний балл/средний скор по этим предметам — relevant.
ВАЖНО: запросы про среднее по ВСЕМ предметам/курсам/дисциплинам — тоже relevant,
даже если нет названия конкретного предмета.
ВАЖНО: запросы вида "кто лектор по ...", "лектором по ... является",
"кто ведет лекции по ..." для этих предметов — relevant.
ВАЖНО: запросы про лучших студентов в единственном числе тоже relevant:
"кто из студентов лучший?", "кто лучший студент по ...", "лучший студент по ...".
Примеры relevant:
- "тервер среднее"
- "машинка скор с усреднением"
- "средний балл по всем предметам"
- "кто лектор по дисциплине ml"
- "по теорверу лектор это"
- "кто ведет лекции по оптам"
- "теория оптимизации: кто из студентов лучший?"

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
