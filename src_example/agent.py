"""Student-domain agent that routes user query to src.api functions."""

from __future__ import annotations

import json

from src.api import get_avg_overall_score, get_avg_score, get_top_students, search_rag

from .classify_intent import classify_intent
from .logger import logger
from .router import route_query

IRRELEVANT_MESSAGE = 'Вопрос не релевантен для агента'


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
    return {'answer': json.dumps(search_rag(query=query), ensure_ascii=False)}


def main() -> dict:
    """Run simple one-shot CLI input for local debugging."""
    user_input = input('user: ').strip()
    return agent(user_input)


if __name__ == '__main__':
    print(main())
