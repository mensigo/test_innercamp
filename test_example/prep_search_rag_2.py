"""Run time-oriented search_rag cases and print retrieved chunk previews."""

from __future__ import annotations

from dataclasses import dataclass

import argparse

from src.api import search_rag

TOP_K = 7
PREVIEW_LIMIT = 160
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_RESET = '\033[0m'


@dataclass
class SearchRagCase:
    idx: int
    user_query: str
    expected_answer: str


CASES_ML_TIME: list[SearchRagCase] = [
    SearchRagCase(
        idx=1,
        user_query='когда проходят лекции по мл',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
    SearchRagCase(
        idx=2,
        user_query='время лекций по машинному обучению',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
    SearchRagCase(
        idx=3,
        user_query='лекции по машинке проходят время',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
    SearchRagCase(
        idx=4,
        user_query='расписание лекций по мл',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
    SearchRagCase(
        idx=5,
        user_query='лекционная часть по мл время',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
    SearchRagCase(
        idx=6,
        user_query='в какое время проходят лекции по машинке',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
    SearchRagCase(
        idx=7,
        user_query='в какой день и время лекции по мл',
        expected_answer='по пятницам, 11:10 - 12:30, в ауд. П8а.',
    ),
]

CASES_PROB_TIME: list[SearchRagCase] = [
    SearchRagCase(
        idx=8,
        user_query='когда проходят лекции по теории вероятности',
        expected_answer='| Пятница     | 10:30 – 12:00 | Все      | лекция      | R302',
    ),
    SearchRagCase(
        idx=9,
        user_query='время лекций по теорверу',
        expected_answer='| Пятница     | 10:30 – 12:00 | Все      | лекция      | R302',
    ),
    SearchRagCase(
        idx=10,
        user_query='день и время лекций по теорверу',
        expected_answer='| Пятница     | 10:30 – 12:00 | Все      | лекция      | R302',
    ),
    SearchRagCase(
        idx=11,
        user_query='расписание лекций по вероятности',
        expected_answer='| Пятница     | 10:30 – 12:00 | Все      | лекция      | R302',
    ),
    SearchRagCase(
        idx=12,
        user_query='лекционная часть по вероятности время',
        expected_answer='| Пятница     | 10:30 – 12:00 | Все      | лекция      | R302',
    ),
]

CASES_OPT_TIME: list[SearchRagCase] = [
    SearchRagCase(
        idx=13,
        user_query='когда проходят лекции по оптимизации',
        expected_answer='вторник, лекция в 13:00 (ауд. П9)',
    ),
    SearchRagCase(
        idx=14,
        user_query='время лекций по оптам',
        expected_answer='вторник, лекция в 13:00 (ауд. П9)',
    ),
    SearchRagCase(
        idx=15,
        user_query='оптимизация день и время лекций',
        expected_answer='вторник, лекция в 13:00 (ауд. П9)',
    ),
    SearchRagCase(
        idx=16,
        user_query='расписание лекций по оптимизации',
        expected_answer='вторник, лекция в 13:00 (ауд. П9)',
    ),
    SearchRagCase(
        idx=17,
        user_query='по метоптам когда идут лекции',
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


def print_cases(cases: list[SearchRagCase], title: str, show_full_chunk: bool = False):
    """Run search_rag for each case and print chunk previews."""
    print(title)
    print(f'using top_k={TOP_K}\n')

    for case in cases:
        print(f'[{case.idx}] query: {case.user_query}')
        print('    ------------------------------------')
        result = search_rag(query=case.user_query, k=TOP_K)
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
        CASES_ML_TIME,
        'search_rag ml (time)',
        show_full_chunk=args.print,
    )
    print_cases(
        CASES_PROB_TIME,
        'search_rag prob (time)',
        show_full_chunk=args.print,
    )
    print_cases(
        CASES_OPT_TIME,
        'search_rag opt (time)',
        show_full_chunk=args.print,
    )
