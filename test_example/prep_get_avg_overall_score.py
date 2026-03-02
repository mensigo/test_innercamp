"""Prep hardcoded get_avg_overall_score e2e data and validate against students.csv."""

from __future__ import annotations

from dataclasses import dataclass

import argparse
import pandas as pd

from src.prepare_data import ensure_students_csv


@dataclass
class AvgOverallScoreCase:
    idx: int
    user_query: str
    expected_answer: str


CASES_SIMPLE: list[AvgOverallScoreCase] = [
    AvgOverallScoreCase(
        idx=1,
        user_query='средний балл по всем предметам',
        expected_answer='4.0',
    ),
    AvgOverallScoreCase(
        idx=2,
        user_query='по предметам среднее',
        expected_answer='4.0',
    ),
    AvgOverallScoreCase(
        idx=3,
        user_query='среднее по курсам',
        expected_answer='4.0',
    ),
    AvgOverallScoreCase(
        idx=4,
        user_query='усредненный балл по всем курсам',
        expected_answer='4.0',
    ),
    AvgOverallScoreCase(
        idx=5,
        user_query='все предметы, средний балл',
        expected_answer='4.0',
    ),
]

CASES_COMPLEX: list[AvgOverallScoreCase] = [
    AvgOverallScoreCase(
        idx=6,
        user_query='средний скор по предметам',
        expected_answer='4.0',
    ),
    AvgOverallScoreCase(
        idx=7,
        user_query='скор усредненный по курсам',
        expected_answer='4.0',
    ),
    AvgOverallScoreCase(
        idx=8,
        user_query='по дисциплинам всем среднее',
        expected_answer='4.0',
    ),
]

STUDENTS_CSV_PATH = ensure_students_csv(force=False)
STUDENTS_DF = pd.read_csv(STUDENTS_CSV_PATH)


if STUDENTS_DF.empty:
    ACTUAL_ANSWER = ''
else:
    SCORES = STUDENTS_DF['score'].astype(float).round(1)
    AVG_SCORE = round(float(SCORES.mean()), 1)
    ACTUAL_ANSWER = f'{AVG_SCORE:.1f}'


def validate_cases(
    cases: list[AvgOverallScoreCase],
    desc: str = 'validating get_avg_overall_score',
):
    """Validate case metadata and expected answers against generated dataset."""
    print(desc)
    ok = True
    actual_answer = ACTUAL_ANSWER
    for case in cases:
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

    ok_simple = validate_cases(CASES_SIMPLE, 'validating get_avg_overall_score simple')
    if not ok_simple:
        print('ERROR: get_avg_overall_score simple cases are not valid')
        exit(1)
    print('...ok\n')

    ok_complex = validate_cases(
        CASES_COMPLEX, 'validating get_avg_overall_score complex'
    )
    if not ok_complex:
        print('ERROR: get_avg_overall_score complex cases are not valid')
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
