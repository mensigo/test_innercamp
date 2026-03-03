"""Run place-oriented vector_search cases and print retrieved chunk previews."""

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


CASES_ML_PLACE: list[VectorSearchCase] = [
    VectorSearchCase(
        idx=1,
        user_query='где проходят лекции по мл',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
    VectorSearchCase(
        idx=2,
        user_query='аудитория лекций по машинному обучению',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
    VectorSearchCase(
        idx=3,
        user_query='лекции по машинке место',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
    VectorSearchCase(
        idx=4,
        user_query='расписание лекций по мл, какая аудитория',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
]

CASES_PROB_PLACE: list[VectorSearchCase] = [
    VectorSearchCase(
        idx=5,
        user_query='где проходят лекции по теории вероятности',
        expected_answer='| Пятница     | 10:30 – 12:00 | Все      | лекция      | R302',
    ),
    VectorSearchCase(
        idx=6,
        user_query='место лекций по теорверу',
        expected_answer='| Пятница     | 10:30 – 12:00 | Все      | лекция      | R302',
    ),
    VectorSearchCase(
        idx=7,
        user_query='по расписанию лекции по теорверу проходят в ?',
        expected_answer='| Пятница     | 10:30 – 12:00 | Все      | лекция      | R302',
    ),
    VectorSearchCase(
        idx=8,
        user_query='по вероятности лекции где',
        expected_answer='| Пятница     | 10:30 – 12:00 | Все      | лекция      | R302',
    ),
]

CASES_OPT_PLACE: list[VectorSearchCase] = [
    VectorSearchCase(
        idx=9,
        user_query='где проходят лекции по оптимизации',
        expected_answer='вторник, лекция в 13:00 (ауд. П9)',
    ),
    VectorSearchCase(
        idx=10,
        user_query='ауд лекций по оптам',
        expected_answer='вторник, лекция в 13:00 (ауд. П9)',
    ),
    VectorSearchCase(
        idx=11,
        user_query='оптимизация аудитория лекций',
        expected_answer='вторник, лекция в 13:00 (ауд. П9)',
    ),
    VectorSearchCase(
        idx=12,
        user_query='лекции по оптимизации проходят в аудитории номер ?',
        expected_answer='вторник, лекция в 13:00 (ауд. П9)',
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
        CASES_ML_PLACE,
        'vector_search ml (place)',
        show_full_chunk=args.print,
    )
    print_cases(
        CASES_PROB_PLACE,
        'vector_search prob (place)',
        show_full_chunk=args.print,
    )
    print_cases(
        CASES_OPT_PLACE,
        'vector_search opt (place)',
        show_full_chunk=args.print,
    )
