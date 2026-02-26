"""LLM-based query routing for src_learn agent."""

from __future__ import annotations

import json

from src_learn.config import config
from src_learn.logger import logger
from src_learn.utils import post_chat_completions


def route_query(user_query: str) -> dict:
    """Route query to tool + operation via LLM function calling."""
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'route_query',
                'description': 'Select a local tool and operation for student-domain query.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'tool_name': {
                            'type': 'string',
                            'enum': [
                                'database_tool',
                                'rag_tool',
                                'student_meta_tool',
                            ],
                        },
                        'operation': {
                            'type': 'string',
                            'enum': [
                                'top_students',
                                'hardest_questions',
                                'failure_stats',
                                'student_meta',
                                'rag_answer',
                            ],
                        },
                        'subject_name': {'type': 'string'},
                        'student_name': {'type': 'string'},
                        'top_k': {'type': 'integer'},
                        'question': {'type': 'string'},
                    },
                    'required': ['tool_name', 'operation'],
                },
            },
        }
    ]

    system_prompt = """
Ты маршрутизируешь запросы для локального агента студенческой аналитики.
Выбери ровно одну пару tool+operation и верни только function call.

КРИТИЧЕСКОЕ ПРАВИЛО (высший приоритет):
Если в запросе есть намерение выбрать лучших студентов по предмету,
ВСЕГДА выбирай:
- tool_name = "database_tool"
- operation = "top_students"

Это включает формулировки:
- "лучшие студенты", "лучший студент", "кто лучший", "кто лучшие"
- "топ студентов", "топ учащихся", "рейтинг студентов", "лидеры"
- перестановки слов: "среди студентов лучшие по ...", "...: лучшие студенты"

ЖЕСТКИЙ ЗАПРЕТ:
- Для запросов про лучших/топ студентов НИКОГДА не выбирай "rag_tool".

Точное соответствие типов запросов:
- Лучшие студенты по предмету -> database_tool + top_students
- Самые сложные вопросы по предмету -> database_tool + hardest_questions
- Количество неуспешных (более 80% не сдали) -> database_tool + failure_stats
- Вопросы, где конкретный студент не справился, и его meta-оценки -> student_meta_tool + student_meta
- Теоретический вопрос про тему/концепт -> rag_tool + rag_answer

Чеклист перед выбором инструмента:
1) Есть ли в запросе intent "лучшие/топ/рейтинг/лидеры" + "студенты/учащиеся"?
   -> сразу database_tool + top_students.
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

Для top_students:
- Если top_k явно указан в запросе, верни его.
- Если не указан, верни top_k = 3.
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
            'tool_name': 'rag_tool',
            'operation': 'rag_answer',
            'question': user_query,
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
            'tool_name': 'database_tool',
            'operation': 'top_students',
            'subject_name': user_query,
            'top_k': 3,
        }
    if 'hardest' in lowered:
        return {
            'tool_name': 'database_tool',
            'operation': 'hardest_questions',
            'subject_name': user_query,
            'top_k': 3,
        }
    return {'tool_name': 'rag_tool', 'operation': 'rag_answer', 'question': user_query}
