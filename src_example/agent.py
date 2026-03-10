"""Student-domain agent that routes user query to src.api functions."""

from __future__ import annotations

from src.api import vector_search
from src.api import get_avg_overall_score, get_avg_score, get_top_students

from .answer_with_rag import answer_with_rag
from .classify_intent import classify_intent
from .classify_intent_vector_search import classify_intent_vector_search
from .extract_lecturer import extract_lecturer
from .extract_location import extract_location
from .extract_schedule import extract_schedule
from .logger import logger
from .router import route_query

IRRELEVANT_MESSAGE = 'Вопрос не релевантен для агента'

RAG_TOP_K = 2


def _format_top_students(rows: list[dict]) -> str:
    """Format top students rows into a compact one-line answer."""
    chunks: list[str] = []
    for row in rows:
        name = str(row.get('name') or '').strip()
        score = row.get('score')
        if not name or score is None:
            continue
        chunks.append(f'{name} ({float(score):.1f})')
    return ', '.join(chunks)


def _extract_k(route: dict) -> int:
    """Extract and validate k from route payload."""
    raw_k = route.get('k', route.get('top_k', 3))
    try:
        return int(raw_k)
    except (TypeError, ValueError):
        return 3


def _build_vector_query(user_query: str, vector_intent: dict) -> str:
    """Build a concise retrieval query for vector search."""
    intent_type = str(vector_intent.get('intent_type') or '')
    subject_name = str(vector_intent.get('subject_name') or '').strip()
    if not subject_name:
        return user_query

    subject_label = {
        'Machine Learning': 'Машинное обучение',
        'Probability Theory': 'Теория вероятности',
        'Optimization Theory': 'Методы оптимизации',
    }.get(subject_name, subject_name)

    if intent_type == 'lecturer_name':
        if subject_name == 'Machine Learning':
            return 'Машинное обучение ПМИ ФКН ВШЭ О курсе Лектор'
        if subject_name == 'Probability Theory':
            return 'Теория вероятности Лектор'
        if subject_name == 'Optimization Theory':
            return 'Методы оптимизации Преподаватели лектор'
        return f'{subject_label} курс лекции лектор преподаватель'

    if intent_type == 'lecture_schedule':
        if subject_name == 'Machine Learning':
            return 'Машинное обучение О курсе Семинары Правила выставления оценок'
        if subject_name == 'Probability Theory':
            return 'Теория вероятности Расписание занятий Тип занятия Аудитория Преподаватель'
        if subject_name == 'Optimization Theory':
            return 'Методы оптимизации Группа Расписание Инвайт anytask'
        return f'{subject_label} курс лекции расписание день время'

    if intent_type == 'lecture_location':
        if subject_name == 'Machine Learning':
            return 'Машинное обучение О курсе лекции аудитория место проведения'
        if subject_name == 'Probability Theory':
            return (
                'Теория вероятности Расписание занятий День недели Время Группа '
                'Тип занятия Аудитория Преподаватель Семинарист'
            )
        if subject_name == 'Optimization Theory':
            return 'Методы оптимизации лекции аудитория место проведения'
        return f'{subject_label} курс лекции аудитория место'

    if intent_type == 'books_for_course':
        return f'{subject_label} курс литература книги учебники'

    return f'{subject_label} курс'


def agent(user_query: str) -> dict:
    """Run classify->route->API execution and return answer payload."""
    if not classify_intent(user_query):
        return {'answer': IRRELEVANT_MESSAGE}

    route = route_query(user_query)
    logger.info(f'agent // route: {route}')

    tool_name = str(route.get('tool_name') or '')
    if tool_name == 'get_top_students':
        subject_name = str(route.get('subject_name') or '').strip()
        try:
            rows = get_top_students(subject_name=subject_name, k=_extract_k(route))
        except ValueError as ex:
            return {'answer': str(ex)}
        return {'answer': _format_top_students(rows)}

    if tool_name == 'get_avg_score':
        subject_name = str(route.get('subject_name') or '').strip()
        result = get_avg_score(subject_name)
        avg_score = result.get('avg_score')
        if avg_score is None:
            return {'answer': ''}
        return {'answer': f'{float(avg_score):.1f}'}

    if tool_name == 'get_avg_overall_score':
        result = get_avg_overall_score()
        avg_score = result.get('avg_score')
        if avg_score is None:
            return {'answer': ''}
        return {'answer': f'{float(avg_score):.1f}'}

    query = str(route.get('query') or user_query).strip()
    vector_intent = classify_intent_vector_search(user_query)
    logger.info(f'agent // vector_intent: {vector_intent}')

    retrieval_query = _build_vector_query(query, vector_intent)
    if str(vector_intent.get('intent_type') or '') == 'lecturer_name':
        subject_name = str(vector_intent.get('subject_name') or '').strip()
        rag_result = vector_search(query=retrieval_query, k=3)['chunks']
        chunks = [
            chunk for chunk in rag_result if isinstance(chunk, str) and chunk.strip()
        ]
        answer = extract_lecturer(
            user_query=user_query, subject_name=subject_name, chunks=chunks
        )
        return {'answer': answer}

    if str(vector_intent.get('intent_type') or '') == 'lecture_schedule':
        subject_name = str(vector_intent.get('subject_name') or '').strip()
        rag_result = vector_search(query=retrieval_query, k=3).get('chunks', [])
        chunks = [
            chunk for chunk in rag_result if isinstance(chunk, str) and chunk.strip()
        ]
        answer = extract_schedule(
            user_query=user_query, subject_name=subject_name, chunks=chunks
        )
        answer = answer.replace(' – ', ' - ').replace('–', '-')
        return {'answer': answer}

    if str(vector_intent.get('intent_type') or '') == 'lecture_location':
        subject_name = str(vector_intent.get('subject_name') or '').strip()
        rag_result = vector_search(query=retrieval_query, k=3).get('chunks', [])
        chunks = [
            chunk for chunk in rag_result if isinstance(chunk, str) and chunk.strip()
        ]
        answer = extract_location(
            user_query=user_query, subject_name=subject_name, chunks=chunks
        )
        return {'answer': answer}

    return {
        'answer': answer_with_rag(
            user_query=user_query, retrieval_query=retrieval_query, top_k=RAG_TOP_K
        )
    }


def main() -> dict:
    """Run simple one-shot CLI input for local debugging."""
    user_input = input('user: ').strip()
    return agent(user_input)


if __name__ == '__main__':
    print(main())
