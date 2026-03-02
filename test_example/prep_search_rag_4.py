"""Run books-oriented search_rag cases and print retrieved chunk previews."""

from __future__ import annotations

from dataclasses import dataclass

import argparse

from src.api import search_rag

TOP_K = 7
PREVIEW_LIMIT = 160
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_RESET = '\033[0m'


ML_BOOKS = """
Книги
- Hastie T., Tibshirani R, Friedman J. The Elements of Statistical Learning (2nd edition). Springer, 2009.
- Bishop C. M. Pattern Recognition and Machine Learning. Springer, 2006.
- Mohri M., Rostamizadeh A., Talwalkar A. Foundations of Machine Learning. MIT Press, 2012.
- Murphy K. Machine Learning: A Probabilistic Perspective. MIT Press, 2012.
- Mohammed J. Zaki, Wagner Meira Jr. Data Mining and Analysis. Fundamental Concepts and Algorithms. Cambridge University Press, 2014.
- Willi Richert, Luis Pedro Coelho. Building Machine Learning Systems with Python. Packt Publishing, 2013.
""".strip()

PROB_BOOKS = """
Литература

Гнеденко Б. В. Курс теории вероятностей. М.: Наука, 1988. [Прим. 1] 
Колмогоров А. Н. Основные понятия теории вероятностей. М.: Наука, 1974. [Прим. 1] 
Феллер В. Введение в теорию вероятностей и её приложения, в 2-х томах. М.: Мир, 1984. [Прим. 1] 
Боровков А. А. Теория вероятностей. М.: Наука, 1976. [Прим. 1] 
Розанов Ю. А. Теория вероятностей, случайные процессы и математическая статистика. М.: Наука, 1985. [Прим. 1] 
Прохоров Ю. В., Прохоров А. В. Курс лекций по теории вероятностей и математической статистике. М.: МЦНМО, 2019. 
Семаков С. Л. Элементы теории вероятностей и случайных процессов. М.: Физматлит, 2011. 
""".strip()

OPT_BOOKS = """
Рекомендуемая литература

1. J. Nocedal, S. Wright. Numerical optimization. Springer, 2006.
2. S. Boyd, L. Vandenberghe. Convex optimization. Cambridge University Press, 2004.
""".strip()


@dataclass
class SearchRagCase:
    idx: int
    user_query: str
    expected_answer: str


CASES_ML_BOOKS: list[SearchRagCase] = [
    SearchRagCase(
        idx=1,
        user_query='какие книги почитать по мл',
        expected_answer=ML_BOOKS,
    ),
    SearchRagCase(
        idx=2,
        user_query='литература по машинному обучению',
        expected_answer=ML_BOOKS,
    ),
    SearchRagCase(
        idx=3,
        user_query='книжки по машинке',
        expected_answer=ML_BOOKS,
    ),
    SearchRagCase(
        idx=4,
        user_query='что можно почитать по мл',
        expected_answer=ML_BOOKS,
    ),
]

CASES_PROB_BOOKS: list[SearchRagCase] = [
    SearchRagCase(
        idx=5,
        user_query='литература по теории вероятности',
        expected_answer=PROB_BOOKS,
    ),
    SearchRagCase(
        idx=6,
        user_query='по теорверу книги',
        expected_answer=PROB_BOOKS,
    ),
    SearchRagCase(
        idx=7,
        user_query='книги для ознакомления с курсом теорвера',
        expected_answer=PROB_BOOKS,
    ),
    SearchRagCase(
        idx=8,
        user_query='по теорверу рекомендуемая литература',
        expected_answer=PROB_BOOKS,
    ),
]

CASES_OPT_BOOKS: list[SearchRagCase] = [
    SearchRagCase(
        idx=9,
        user_query='что можно почитать по курсу оптимизации',
        expected_answer=OPT_BOOKS,
    ),
    SearchRagCase(
        idx=10,
        user_query='книги по оптам',
        expected_answer=OPT_BOOKS,
    ),
    SearchRagCase(
        idx=11,
        user_query='оптимизация список книжек',
        expected_answer=OPT_BOOKS,
    ),
    SearchRagCase(
        idx=12,
        user_query='по оптимизации список рекомендованных произведений',
        expected_answer=OPT_BOOKS,
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
        CASES_ML_BOOKS,
        'search_rag ml (books)',
        show_full_chunk=args.print,
    )
    print_cases(
        CASES_PROB_BOOKS,
        'search_rag prob (books)',
        show_full_chunk=args.print,
    )
    print_cases(
        CASES_OPT_BOOKS,
        'search_rag opt (books)',
        show_full_chunk=args.print,
    )
