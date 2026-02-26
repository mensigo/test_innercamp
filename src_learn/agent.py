"""Local student-domain agent with classify and tool orchestration."""

from __future__ import annotations

import json

from src_learn.classify_intent import classify_intent
from src_learn.data_store import ensure_data_files, normalize_student_name
from src_learn.logger import logger
from src_learn.router import route_query
from src_learn.tools import database_tool, rag_tool, student_meta_tool

IRRELEVANT_MESSAGE = 'Irrelevant query, reformulate'


def _extract_last_user_text(message_history: list[dict]) -> str:
    """Extract last user content from history."""
    for message in reversed(message_history):
        if message.get('role') == 'user':
            return str(message.get('content', '')).strip()
    return ''


def _extract_subject(route: dict, user_query: str) -> str:
    """Extract a likely subject string from route and query."""
    subject = str(route.get('subject_name') or '').strip()
    if subject:
        return subject

    lowered = user_query.lower()
    if 'machine learning' in lowered or ' ml' in lowered:
        return 'Machine Learning'
    if 'probability theory' in lowered or 'probability' in lowered:
        return 'Probability Theory'
    if 'optimization theory' in lowered or 'optimization' in lowered:
        return 'Optimization Theory'
    return 'Machine Learning'


def _format_top_students(rows: list[dict]) -> str:
    """Format top students rows into a compact one-line answer."""
    chunks: list[str] = []
    for row in rows:
        name = str(row.get('name') or '').strip()
        grade = row.get('grade')
        if not name or grade is None:
            continue
        chunks.append(f'{name} ({grade})')
    return ', '.join(chunks)


def agent(message_history: list[dict]) -> dict:
    """Run classify->route->tool execution and return minimal answer."""
    ensure_data_files()
    user_query = _extract_last_user_text(message_history)
    if not classify_intent(user_query):
        return {'answer': IRRELEVANT_MESSAGE}

    route = route_query(user_query)
    logger.info(f'agent // route: {route}')

    tool_name = str(route.get('tool_name') or '')
    operation = str(route.get('operation') or '')
    subject_name = _extract_subject(route, user_query)
    top_k = int(route.get('top_k') or 3)

    if tool_name == 'database_tool':
        output = database_tool(
            operation=operation,
            subject_name=subject_name,
            top_k=top_k,
        )
        if 'error' in output:
            return {'answer': output['error']}
        if operation == 'failure_stats':
            count = output['data']['count']
            return {'answer': f'count: {count}'}
        if operation == 'top_students':
            rows = output.get('data') or []
            return {'answer': _format_top_students(rows)}
        return {'answer': json.dumps(output['data'], ensure_ascii=False)}

    if tool_name == 'student_meta_tool':
        student_name = str(route.get('student_name') or '')
        output = student_meta_tool(student_name=student_name, subject_name=subject_name)
        if 'error' in output:
            return {'answer': output['error']}
        return {'answer': json.dumps(output, ensure_ascii=False)}

    # default to rag
    question = str(route.get('question') or user_query)
    output = rag_tool(question)
    if 'error' in output:
        return {'answer': output['error']}
    return {'answer': output['answer']}


def main() -> dict:
    """Run simple one-shot CLI input for local debugging."""
    user_input = input('user: ').strip()
    return agent([{'role': 'user', 'content': user_input}])


if __name__ == '__main__':
    print(main())
