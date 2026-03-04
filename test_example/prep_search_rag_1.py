"""Run lector-oriented vector_search cases and print retrieved chunk previews."""

from __future__ import annotations

from dataclasses import dataclass

import argparse

from src.api import vector_search

TOP_K = 7
PREVIEW_LIMIT = 160
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_RESET = '\033[0m'


@dataclass
class VectorSearchCase:
    idx: int
    user_query: str
    expected_answer: str


CASES_SIMPLE_LECTURER: list[VectorSearchCase] = [
    VectorSearchCase(
        idx=1,
        user_query='лектор по мл',
        expected_answer='Соколов Евгений Андреевич',
    ),
    VectorSearchCase(
        idx=2,
        user_query='лектор по машинному обучению',
        expected_answer='Соколов Евгений Андреевич',
    ),
    VectorSearchCase(
        idx=3,
        user_query='лектор по машинке',
        expected_answer='Соколов Евгений Андреевич',
    ),
    VectorSearchCase(
        idx=4,
        user_query='лектор по теории вероятностей',
        expected_answer='Семаков Сергей Львович',
    ),
    VectorSearchCase(
        idx=5,
        user_query='лектор по теорверу',
        expected_answer='Семаков Сергей Львович',
    ),
    VectorSearchCase(
        idx=6,
        user_query='лектор по вероятностям',
        expected_answer='Семаков Сергей Львович',
    ),
    VectorSearchCase(
        idx=7,
        user_query='лектор по теории оптимизации',
        expected_answer='Кропотов Дмитрий Александрович',
    ),
    VectorSearchCase(
        idx=8,
        user_query='лектор по оптимизации',
        expected_answer='Кропотов Дмитрий Александрович',
    ),
    VectorSearchCase(
        idx=9,
        user_query='лектор по метоптам',
        expected_answer='Кропотов Дмитрий Александрович',
    ),
]

CASES_COMPLEX_LECTURER: list[VectorSearchCase] = [
    VectorSearchCase(
        idx=10,
        user_query='кто лектор по дисциплине ml',
        expected_answer='Соколов Евгений Андреевич',
    ),
    VectorSearchCase(
        idx=11,
        user_query='лектором по мл является',
        expected_answer='Соколов Евгений Андреевич',
    ),
    VectorSearchCase(
        idx=12,
        user_query='машинка читает лекции кто',
        expected_answer='Соколов Евгений Андреевич',
    ),
    VectorSearchCase(
        idx=13,
        user_query='кто лектор по курсу probability theory',
        expected_answer='Семаков Сергей Львович',
    ),
    VectorSearchCase(
        idx=14,
        user_query='по вероятности ведет лекции кто',
        expected_answer='Семаков Сергей Львович',
    ),
    VectorSearchCase(
        idx=15,
        user_query='по теорверу лектор это',
        expected_answer='Семаков Сергей Львович',
    ),
    VectorSearchCase(
        idx=16,
        user_query='кто ведет лекции по оптам',
        expected_answer='Кропотов Дмитрий Александрович',
    ),
    VectorSearchCase(
        idx=17,
        user_query='кто по части лекций оптимизации',
        expected_answer='Кропотов Дмитрий Александрович',
    ),
    VectorSearchCase(
        idx=18,
        user_query='кто отвечает за лекционную часть оптимизации',
        expected_answer='Кропотов Дмитрий Александрович',
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


def _find_expected_chunk_nums(chunks: list[str], expected_answer: str) -> list[int]:
    """Find all chunk numbers where expected_answer is a substring."""
    if not expected_answer:
        return []

    return [
        idx for idx, chunk in enumerate(chunks, start=1) if expected_answer in chunk
    ]


def _format_chunk_nums(chunk_nums: list[int]) -> str:
    """Format matched chunk numbers with color highlighting."""
    if not chunk_nums:
        return f'{COLOR_RED}None{COLOR_RESET}'

    first_num = f'{COLOR_GREEN}{chunk_nums[0]}{COLOR_RESET}'
    if len(chunk_nums) == 1:
        return first_num

    other_nums = ','.join(str(num) for num in chunk_nums[1:])
    return f'{first_num},{other_nums}'


def print_cases(
    cases: list[VectorSearchCase], title: str, show_full_chunk: bool = False
):
    """Run vector_search for each case and print chunk previews."""
    print(title)
    print(f'using top_k={TOP_K}\n')

    for case in cases:
        print(f'[{case.idx}] query: {case.user_query}')
        print('    ------------------------------------')
        result = vector_search(query=case.user_query, k=TOP_K)
        chunks = result['chunks']
        print(f'    retrieved_chunks: {len(chunks)}')
        expected_chunk_nums = _find_expected_chunk_nums(
            chunks=chunks, expected_answer=case.expected_answer
        )
        expected_chunk_nums_str = _format_chunk_nums(expected_chunk_nums)
        print(f'    expected_answer_chunk_nums: {expected_chunk_nums_str}')

        if not chunks:
            print('    - no chunks returned')
            print()
            continue

        for idx, chunk_text in enumerate(chunks, start=1):
            first_row = _shorten(_first_row(chunk_text))
            first_row = first_row or '<empty>'
            print(f'    - chunk_{idx} : {first_row}')
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

    print_cases(
        CASES_SIMPLE_LECTURER,
        'vector_search simple (lecturers)',
        show_full_chunk=args.print,
    )
    print_cases(
        CASES_COMPLEX_LECTURER,
        'vector_search complex cases (lecturers)',
        show_full_chunk=args.print,
    )
