"""LLM-based query routing for example agent."""

from __future__ import annotations

import json

from .config import config
from .logger import logger
from .utils import post_chat_completions


def route_query(user_query: str) -> dict:
    """Route query to one src.api function via LLM function calling."""
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'route_query',
                'description': 'Select one src.api function for student-domain query.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'tool_name': {
                            'type': 'string',
                            'enum': [
                                'get_top_students',
                                'get_avg_score',
                                'get_avg_overall_score',
                                'vector_search',
                            ],
                        },
                        'subject_name': {'type': 'string'},
                        'k': {'type': 'integer'},
                        'query': {'type': 'string'},
                    },
                    'required': ['tool_name'],
                },
            },
        }
    ]

    system_prompt = """
Ты маршрутизируешь запросы для локального агента студенческой аналитики.
Выбери ровно один tool_name (это имя API-функции) и верни только function call.

КРИТИЧЕСКОЕ ПРАВИЛО (высший приоритет):
Если в запросе есть намерение выбрать лучших студентов по предмету,
ВСЕГДА выбирай:
- tool_name = "get_top_students"

ОТДЕЛЬНОЕ КРИТИЧЕСКОЕ ПРАВИЛО:
Если запрос про лектора, лекционную часть курса, кто ведет или кто читает лекции,
ВСЕГДА выбирай:
- tool_name = "vector_search"
- query = исходный пользовательский запрос

ОТДЕЛЬНОЕ КРИТИЧЕСКОЕ ПРАВИЛО:
Если запрос про расписание/день/время лекций, аудиторию лекций,
или литературу/книги по курсу, ВСЕГДА выбирай:
- tool_name = "vector_search"
- query = исходный пользовательский запрос

Это включает формулировки:
- "лучшие студенты", "лучший студент", "кто лучший", "кто лучшие"
- "топ студентов", "топ учащихся", "рейтинг студентов", "лидеры"
- перестановки слов: "среди студентов лучшие по ...", "...: лучшие студенты"

ЖЕСТКИЙ ЗАПРЕТ:
- Для запросов про лучших/топ студентов НИКОГДА не выбирай "vector_search".
- Для запросов про лектора/лекции НИКОГДА не выбирай "get_top_students".

Точное соответствие типов запросов:
- Лучшие студенты по предмету -> get_top_students (subject_name, k)
- Средний балл по конкретному предмету -> get_avg_score (subject_name)
- Средний балл по всем предметам/студентам -> get_avg_overall_score
- Теоретический вопрос про курс/тему/концепт -> vector_search (query)
- Расписание лекций / аудитория / литература -> vector_search (query)

Чеклист перед выбором инструмента:
1) Есть ли в запросе intent "лучшие/топ/рейтинг/лидеры" + "студенты/учащиеся"?
   -> сразу get_top_students.
2) Только если ответа "да" в п.1 нет, можно рассматривать другие tools.

Нормализация предмета для function call:
- Возвращай subject_name только в каноническом виде:
  * "Machine Learning"
  * "Probability Theory"
  * "Optimization Theory"
- Алиасы:
  * "ml", "мл", "машинка", "машинное обучение", "по машинному обучению" -> "Machine Learning"
  * "теорвер", "тервер", "теория вероятности", "по теории вероятностей" -> "Probability Theory"
  * "опты", "метопты", "методы оптимизации", "по теории оптимизации" -> "Optimization Theory"

Для get_top_students:
- Если k явно указан в запросе, верни его.
- Если не указан, верни k = 3.
"""
    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt.strip()},
            {'role': 'user', 'content': user_query},
        ],
        'tools': tools,
        'tool_choice': 'required',
        'temperature': config.freezing,
    }
    response = post_chat_completions(payload, verbose=config.debug)
    if 'error' in response:
        logger.warning(f'agent // route error: {response["error"]}')
        return {
            'tool_name': 'vector_search',
            'query': user_query,
        }

    message = response.get('choices', [{}])[0].get('message', {})
    function_call = message.get('function_call')
    if function_call:
        raw_args = function_call.get('arguments', '{}')
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            pass

    tool_calls = message.get('tool_calls') or []
    if tool_calls:
        raw_args = tool_calls[0].get('function', {}).get('arguments', '{}')
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            pass

    # Fallback heuristic keeps routing deterministic.
    lowered = user_query.lower()
    top_markers = (
        'top',
        'топ',
        'лучшие',
        'лучший',
        'рейтинг',
        'лидеры',
    )
    student_markers = ('student', 'студент', 'учащ', 'ученик')
    has_top_intent = any(marker in lowered for marker in top_markers)
    has_student_intent = any(marker in lowered for marker in student_markers)
    if has_top_intent and has_student_intent:
        return {
            'tool_name': 'get_top_students',
            'subject_name': user_query,
            'k': 3,
        }
    if 'средн' in lowered and (
        'ml' in lowered
        or 'мл' in lowered
        or 'машин' in lowered
        or 'вероят' in lowered
        or 'теорвер' in lowered
        or 'тервер' in lowered
        or 'оптим' in lowered
    ):
        return {'tool_name': 'get_avg_score', 'subject_name': user_query}

    if 'средн' in lowered:
        return {'tool_name': 'get_avg_overall_score'}

    return {'tool_name': 'vector_search', 'query': user_query}
