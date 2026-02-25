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
Выбери ровно одну пару tool+operation.

Соответствие:
- Лучшие студенты по предмету -> database_tool + top_students
  Примеры формулировок:
  * "покажи лучших студентов"
  * "топ студентов"
  * "кто лучшие студенты"
  * "нужны лучшие студенты"

- Самые сложные вопросы по предмету -> database_tool + hardest_questions
- Количество неуспешных (более 80% не сдали) по предмету -> database_tool + failure_stats
- Вопросы, где студент не справился, и его meta-оценки -> student_meta_tool + student_meta
- Теоретический вопрос / концепт экзамена -> rag_tool + rag_answer

Нормализация предметов для function call:
- Всегда возвращай subject_name в каноническом виде:
  * "Machine Learning"
  * "Probability Theory"
  * "Optimization Theory"
- Алиасы для "Machine Learning": "ml", "мл", "машинка", "машинное обучение"
- Алиасы для "Probability Theory": "теорвер", "тервер", "теория вероятности"
- Алиасы для "Optimization Theory": "опты", "метопты", "методы оптимизации"

Верни только function call.
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
    if 'top' in lowered and 'student' in lowered:
        return {
            'tool_name': 'database_tool',
            'operation': 'top_students',
            'subject_name': user_query,
            'top_k': 5,
        }
    if 'hardest' in lowered:
        return {
            'tool_name': 'database_tool',
            'operation': 'hardest_questions',
            'subject_name': user_query,
            'top_k': 3,
        }
    return {'tool_name': 'rag_tool', 'operation': 'rag_answer', 'question': user_query}
