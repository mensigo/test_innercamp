"""Run search_rag for predefined queries and print chunk previews."""

from __future__ import annotations

from dataclasses import dataclass

import argparse

from src.api import search_rag

TOP_K = 2
PREVIEW_LIMIT = 160


@dataclass
class SearchRagCase:
    idx: int
    tool_name: str
    user_query: str


CASES_SIMPLE: list[SearchRagCase] = [
    SearchRagCase(idx=1, tool_name='search_rag', user_query='лектор по мл'),
    SearchRagCase(
        idx=2, tool_name='search_rag', user_query='лектор по теории вероятностей'
    ),
    SearchRagCase(
        idx=3, tool_name='search_rag', user_query='лектор по методам оптимизации'
    ),
    SearchRagCase(
        idx=4, tool_name='search_rag', user_query='лектор по машинному обучению'
    ),
    SearchRagCase(idx=5, tool_name='search_rag', user_query='лектор по оптимизации'),
]

CASES_COMPLEX: list[SearchRagCase] = [
    SearchRagCase(
        idx=6,
        tool_name='search_rag',
        user_query='кто ведет дисциплину machine learning',
    ),
    SearchRagCase(
        idx=7,
        tool_name='search_rag',
        user_query='кто лектор на курсе теория вероятностей',
    ),
    SearchRagCase(
        idx=8,
        tool_name='search_rag',
        user_query='кто читает лекции по методам оптимизации',
    ),
    SearchRagCase(
        idx=9,
        tool_name='search_rag',
        user_query='подскажи фамилию преподавателя по машинке',
    ),
    SearchRagCase(
        idx=10,
        tool_name='search_rag',
        user_query='кто преподаватель по терверу',
    ),
]


def _first_row(text: str) -> str:
    """Get first non-empty row from chunk text."""
    for row in text.splitlines():
        stripped = row.strip()
        if stripped:
            return stripped
    return ''


def _shorten(text: str, limit: int = PREVIEW_LIMIT) -> str:
    """Shorten preview text for compact CLI output."""
    if len(text) <= limit:
        return text
    return f'{text[: limit - 3]}...'


def _print_cases(cases: list[SearchRagCase], title: str, show_full_chunk: bool = False):
    """Run search_rag for each case and print chunk previews."""
    print(title)
    print(f'using top_k={TOP_K}\n')

    for case in cases:
        print(f'[{case.idx}] query: {case.user_query}')
        print('    ------------------------------------')
        result = search_rag(query=case.user_query, k=TOP_K)
        chunks_raw = result.get('chunks')
        chunks = chunks_raw if isinstance(chunks_raw, list) else []
        print(f'    tool_name: {case.tool_name}')
        print(f'    retrieved_chunks: {len(chunks)}')

        if not chunks:
            print('    - no chunks returned')
            print()
            continue

        for idx, chunk in enumerate(chunks, start=1):
            chunk_text = chunk if isinstance(chunk, str) else str(chunk)
            first_row = _shorten(_first_row(chunk_text))
            first_row = first_row or '<empty>'
            print(f'    - chunk_{idx} first_row: {first_row}')
            if show_full_chunk:
                print(f'      chunk_{idx} full_text:\n{chunk_text}')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--print',
        action='store_true',
        help='print full chunk text in addition to first rows',
    )
    args = parser.parse_args()

    _print_cases(CASES_SIMPLE, 'search_rag simple cases', show_full_chunk=args.print)
    _print_cases(CASES_COMPLEX, 'search_rag complex cases', show_full_chunk=args.print)
