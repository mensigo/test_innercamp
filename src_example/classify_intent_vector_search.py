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
    # if 'ml' in lowered or 'мл' in lowered or 'машин' in lowered or 'машинк' in lowered:
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
    #     return 'lecturer_name'

    # has_location_signal = (
    #     'аудитор' in lowered
    #     or 'в какой аудитории' in lowered
    #     or 'какая аудитория' in lowered
    #     or 'место' in lowered
    #     or 'где' in lowered
    # )
    # has_schedule_signal = (
    #     'распис' in lowered or 'когда' in lowered or 'время' in lowered
    # )
    # if has_location_signal and not has_schedule_signal:
    #     return 'lecture_location'
    # if has_location_signal and has_schedule_signal:
    #     return 'lecture_location'
    # if has_schedule_signal:
    #     return 'lecture_schedule'
    # if 'книг' in lowered or 'литератур' in lowered or 'учебник' in lowered:
    #     return 'books_for_course'
    # return 'other'


def classify_intent_vector_search(user_query: str) -> dict:
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
                                'lecturer_name',
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
- lecturer_name (кто лектор/кто ведет лекции)
- lecture_schedule (когда лекции, расписание, день/время лекций, лекционная часть по времени)
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
Не путай lecture_schedule с lecture_location:
- lecture_schedule: вопросы про день/время лекций ("когда", "во сколько", "в какой день", "время лекций").
- lecture_location: вопросы про место/аудиторию ("где", "в какой аудитории", "какая аудитория", "место лекций").
- Слова "расписание лекций" сами по себе означают lecture_schedule.
- Для lecture_location в запросе обязательно должен быть явный признак места:
  "где", "аудитория", "место", "в какой аудитории", "какая аудитория".
- Если в запросе есть слово "время" (или "во сколько"), выбирай lecture_schedule.
- Формулировки типа "лекции ... проходят время" относятся к lecture_schedule.
- Если в запросе одновременно есть слова о расписании и явный запрос места/аудитории
  (например "по расписанию ... какая аудитория", "по расписанию ... проходят в какой аудитории"),
  выбирай lecture_location.
- Если есть фразы "какая аудитория" или "в какой аудитории", это ВСЕГДА lecture_location.
- Формулировка "лекции ... проходят в ?" без запроса времени относится к lecture_location.
- Пример: "расписание лекций по мл, какая аудитория" -> lecture_location.

Примеры для самопроверки:
- "расписание лекций по мл" -> lecture_schedule
- "расписание лекций по оптимизации" -> lecture_schedule
- "когда проходят лекции по теорверу" -> lecture_schedule
- "где проходят лекции по мл" -> lecture_location
- "аудитория лекций по машинному обучению" -> lecture_location
- "по расписанию лекции по теорверу проходят в ?" -> lecture_location
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
