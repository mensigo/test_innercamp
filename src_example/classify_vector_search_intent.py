"""LLM-based intent extraction for vector_search branch."""

from __future__ import annotations

import json

from .config import config
from .logger import logger
from .utils import post_chat_completions


def _fallback_subject(user_query: str) -> str:
    """Infer canonical subject name with lightweight aliases."""
    return user_query
    # lowered = user_query.lower()
    # if (
    #     'ml' in lowered
    #     or 'мл' in lowered
    #     or 'машин' in lowered
    #     or 'машинк' in lowered
    # ):
    #     return 'Machine Learning'
    # if (
    #     'теорвер' in lowered
    #     or 'тервер' in lowered
    #     or 'вероятн' in lowered
    #     or 'probability' in lowered
    # ):
    #     return 'Probability Theory'
    # if (
    #     'оптим' in lowered
    #     or 'метопт' in lowered
    #     or 'опт' in lowered
    #     or 'optimization' in lowered
    # ):
    #     return 'Optimization Theory'
    # return ''


def _fallback_intent(user_query: str) -> str:
    """Infer coarse vector_search intent from query wording."""
    return user_query
    # lowered = user_query.lower()
    # if (
    #     'лектор' in lowered
    #     or 'кто ведет лекции' in lowered
    #     or 'кто ведёт лекции' in lowered
    #     or 'кто читает лекции' in lowered
    # ):
    #     return 'lector_name'
    # if 'распис' in lowered or 'когда лекц' in lowered or 'время лекц' in lowered:
    #     return 'lecture_schedule'
    # if 'аудитор' in lowered or 'где лекц' in lowered or 'место лекц' in lowered:
    #     return 'lecture_location'
    # if 'книг' in lowered or 'литератур' in lowered or 'учебник' in lowered:
    #     return 'books_for_course'
    # return 'other'


def classify_vector_search_intent(user_query: str) -> dict:
    """Extract intent and canonical subject for vector search requests."""
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'classify_vector_search_intent',
                'description': 'Определи подтип intent и канонический курс.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'intent_type': {
                            'type': 'string',
                            'enum': [
                                'lector_name',
                                'lecture_schedule',
                                'lecture_location',
                                'books_for_course',
                                'other',
                            ],
                        },
                        'subject_name': {'type': 'string'},
                    },
                    'required': ['intent_type'],
                },
            },
        }
    ]

    system_prompt = """
Ты извлекаешь подтип intent для ветки vector_search.
Верни только function call.

Допустимые intent_type:
- lector_name (кто лектор/кто ведет лекции)
- lecture_schedule (когда лекции, расписание)
- lecture_location (где лекции, аудитория)
- books_for_course (книги/литература/учебники курса)
- other

Нормализуй subject_name в канонические названия:
- "Machine Learning"
- "Probability Theory"
- "Optimization Theory"

Алиасы:
- "ml", "мл", "машинка", "машинное обучение" -> "Machine Learning"
- "теорвер", "тервер", "вероятности", "probability theory" -> "Probability Theory"
- "оптимизация", "метопты", "метоптам", "optimization theory" -> "Optimization Theory"

Если предмет не распознан, верни пустой subject_name.
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
        logger.warning(f'agent // vector intent error: {response["error"]}')
        return {
            'intent_type': _fallback_intent(user_query),
            'subject_name': _fallback_subject(user_query),
        }

    message = response.get('choices', [{}])[0].get('message', {})
    function_call = message.get('function_call')
    if function_call:
        raw_args = function_call.get('arguments', '{}')
        try:
            parsed = json.loads(raw_args)
            return {
                'intent_type': str(parsed.get('intent_type') or 'other'),
                'subject_name': str(parsed.get('subject_name') or '').strip(),
            }
        except json.JSONDecodeError:
            pass

    tool_calls = message.get('tool_calls') or []
    if tool_calls:
        raw_args = tool_calls[0].get('function', {}).get('arguments', '{}')
        try:
            parsed = json.loads(raw_args)
            return {
                'intent_type': str(parsed.get('intent_type') or 'other'),
                'subject_name': str(parsed.get('subject_name') or '').strip(),
            }
        except json.JSONDecodeError:
            pass

    return {
        'intent_type': _fallback_intent(user_query),
        'subject_name': _fallback_subject(user_query),
    }
