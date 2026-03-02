"""Prep hardcoded get_top_students e2e data and validate against students.csv."""

from __future__ import annotations

from dataclasses import dataclass

import argparse
import pandas as pd

from src.prepare_data import ensure_students_csv


@dataclass
class AvgScoreCase:
    idx: int
    tool_name: str
    subject_name: str
    user_query: str
    expected_answer: str


CASES_SIMPLE: list[AvgScoreCase] = [
    AvgScoreCase(
        idx=1,
        tool_name='get_avg_score',
        subject_name='Machine Learning',
        user_query='средний балл по машинному обучению',
        expected_answer='4.1',
    ),
    AvgScoreCase(
        idx=2,
        tool_name='get_avg_score',
        subject_name='Probability Theory',
        user_query='средний скор по теории вероятности',
        expected_answer='3.9',
    ),
    AvgScoreCase(
        idx=3,
        tool_name='get_avg_score',
        subject_name='Optimization Theory',
        user_query='усредненный скор по оптимизации',
        expected_answer='4.0',
    ),
    AvgScoreCase(
        idx=4,
        tool_name='get_avg_score',
        subject_name='Probability Theory',
        user_query='по теории вероятности средняя оценка',
        expected_answer='3.9',
    ),
    AvgScoreCase(
        idx=5,
        tool_name='get_avg_score',
        subject_name='Optimization Theory',
        user_query='покажи среднее по методам оптимизации',
        expected_answer='4.0',
    ),
]

CASES_COMPLEX: list[AvgScoreCase] = [
    AvgScoreCase(
        idx=6,
        tool_name='get_avg_score',
        subject_name='Machine Learning',
        user_query='среднее по мл',
        expected_answer='4.1',
    ),
    AvgScoreCase(
        idx=7,
        tool_name='get_avg_score',
        subject_name='Optimization Theory',
        user_query='средний балл, опты',
        expected_answer='4.0',
    ),
    AvgScoreCase(
        idx=8,
        tool_name='get_avg_score',
        subject_name='Probability Theory',
        user_query='тервер среднее',
        expected_answer='3.9',
    ),
    AvgScoreCase(
        idx=9,
        tool_name='get_avg_score',
        subject_name='Machine Learning',
        user_query='машинка скор с усреднением',
        expected_answer='4.1',
    ),
    AvgScoreCase(
        idx=10,
        tool_name='get_avg_score',
        subject_name='Optimization Theory',
        user_query='выведи балл как среднее по метоптам',
        expected_answer='4.0',
    ),
]

_STUDENTS_CSV_PATH = ensure_students_csv(force=False)
_STUDENTS_DF = pd.read_csv(_STUDENTS_CSV_PATH)


def _expected_answer(subject_name: str) -> str:
    filtered = _STUDENTS_DF[_STUDENTS_DF['subject_name'] == subject_name].copy()
    if filtered.empty:
        return ''

    filtered['score'] = filtered['score'].astype(float).round(1)
    avg_score = round(float(filtered['score'].mean()), 1)
    return f'{avg_score:.1f}'


def validate_cases(cases: list[AvgScoreCase], desc: str = 'validating get_avg_score'):
    """Validate case metadata and expected answers against generated dataset."""
    print(desc)
    ok = True
    known_subjects = set(_STUDENTS_DF['subject_name'])
    for case in cases:
        if case.tool_name != 'get_avg_score':
            ok = False
            print(f'invalid tool_name for idx={case.idx}: {case.tool_name}')
        if case.subject_name not in known_subjects:
            ok = False
            print(f'unknown subject for idx={case.idx}: {case.subject_name}')

        actual_answer = _expected_answer(case.subject_name)
        if case.expected_answer != actual_answer:
            ok = False
            print(f'expected_answer mismatch for idx={case.idx}')
            print(f'query:     {case.user_query}')
            print(f'expected:  {case.expected_answer}')
            print(f'computed:  {actual_answer}')
            print()
    return ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--print', action='store_true', help='print detailed output'
    )
    args = parser.parse_args()

    ok_simple = validate_cases(CASES_SIMPLE, 'validating get_avg_score simple')
    if not ok_simple:
        print('ERROR: get_avg_score simple cases are not valid')
        exit(1)
    print('...ok\n')

    ok_complex = validate_cases(CASES_COMPLEX, 'validating get_avg_score complex')
    if not ok_complex:
        print('ERROR: get_avg_score complex cases are not valid')
        exit(1)
    print('...ok\n')

    if args.print:
        print('\nCASES_SIMPLE:\n')
        for case in CASES_SIMPLE:
            print(f"\t\t('{case.user_query}',")
            print(f"\t\t'{case.expected_answer}')")
        print('\nCASES_COMPLEX:\n')
        for case in CASES_COMPLEX:
            print(f"\t\t('{case.user_query}',")
            print(f"\t\t'{case.expected_answer}')")
