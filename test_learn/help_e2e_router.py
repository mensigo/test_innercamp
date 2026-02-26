"""Helper script to inspect top_students outputs for e2e router queries."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from src_learn.tools.database_tool import database_tool

DEFAULT_TOP_K = 3
E2E_ROUTER_TEST_FILE = Path(__file__).with_name('test_e2e_router.py')
TOP_K_WORDS = {
    'один': 1,
    'одна': 1,
    'два': 2,
    'две': 2,
    'три': 3,
    'четыре': 4,
    'пять': 5,
    'шесть': 6,
    'семь': 7,
    'восемь': 8,
    'девять': 9,
    'десять': 10,
}


def _extract_parametrize_queries(file_path: Path) -> list[str]:
    """Read query values from pytest.mark.parametrize decorators."""
    tree = ast.parse(file_path.read_text(encoding='utf-8'))
    queries: list[str] = []

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            marker = decorator.func
            if (
                not isinstance(marker, ast.Attribute)
                or marker.attr != 'parametrize'
                or len(decorator.args) < 2
            ):
                continue
            param_name = ast.literal_eval(decorator.args[0])
            if param_name != 'query':
                continue
            query_values = ast.literal_eval(decorator.args[1])
            if isinstance(query_values, list):
                for value in query_values:
                    if isinstance(value, str):
                        queries.append(value)
    return queries


def _subject_from_query(query: str) -> str:
    """Map a free-form query to canonical subject name."""
    lowered = query.lower()

    if (
        'машин' in lowered
        or 'machine learning' in lowered
        or ' ml' in f' {lowered}'
        or 'мл' in lowered
        or 'машинке' in lowered
    ):
        return 'Machine Learning'

    if (
        'вероятност' in lowered
        or 'probability' in lowered
        or 'теорвер' in lowered
        or 'тервер' in lowered
    ):
        return 'Probability Theory'

    if (
        'оптимизац' in lowered
        or 'optimization' in lowered
        or 'оптам' in lowered
        or 'метоптам' in lowered
        or 'методам оптимизации' in lowered
    ):
        return 'Optimization Theory'

    return 'Machine Learning'


def _top_k_from_query(query: str) -> int:
    """Extract top_k from query text, fallback to default."""
    lowered = query.lower()
    patterns = [
        r'топ\s*-\s*(\d+)',
        r'топ\s+(\d+)',
        r'топ(\d+)',
        r'top\s*-\s*(\d+)',
        r'top\s+(\d+)',
        r'top(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        raw_value = int(match.group(1))
        return max(1, min(raw_value, 30))

    for word, value in TOP_K_WORDS.items():
        if re.search(rf'\bтоп\w*\s+{word}\b', lowered):
            return value
        if re.search(rf'\b{word}\s+лучш\w+\s+студент\w*\b', lowered):
            return value
        if re.search(rf'\b{word}\s+учащ\w*\b', lowered):
            return value
    return DEFAULT_TOP_K


def _print_result_for_query(query: str):
    """Print database_tool output for selected query."""
    subject_name = _subject_from_query(query)
    top_k = _top_k_from_query(query)
    result = database_tool(
        operation='top_students',
        subject_name=subject_name,
        top_k=top_k,
    )
    rows = result.get('data') or []
    formatted = ', '.join(
        f'{row["name"]} ({row["grade"]})'
        for row in rows
        if row.get('name') and row.get('grade') is not None
    )
    print(f'query: {query}')
    print(f'subject_name: {subject_name}; top_k: {top_k}')
    print(f'answer: {formatted}')
    print()


def main():
    """Run helper for all queries from test_e2e_router.py."""
    queries = _extract_parametrize_queries(E2E_ROUTER_TEST_FILE)
    if not queries:
        print('No query values found in parametrization.')
        return

    seen: set[str] = set()
    unique_queries: list[str] = []
    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        unique_queries.append(query)

    print(
        f'Loaded {len(unique_queries)} unique queries from {E2E_ROUTER_TEST_FILE.name}'
    )
    print()
    for query in unique_queries:
        _print_result_for_query(query)


if __name__ == '__main__':
    main()
