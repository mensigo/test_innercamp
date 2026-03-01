"""Prep hardcoded top_students e2e data and validate against CSVs."""

from __future__ import annotations

from dataclasses import dataclass

import argparse
import pandas as pd
from tqdm import tqdm

from src_example.data_store import ENROLLMENTS_CSV, META_GRADES_CSV


@dataclass
class TopStudentsCase:
    idx: int
    operation: str
    subject_name: str
    top_k: int
    user_query: str
    expected_answer: str


CASES_SIMPLE: list[TopStudentsCase] = [
    TopStudentsCase(
        idx=1,
        operation='top_students',
        subject_name='Machine Learning',
        top_k=3,
        user_query='машинное обучение: лучшие студенты',
        expected_answer='Avery (80.4), John (76.9), Ethan (76.5)',
    ),
    TopStudentsCase(
        idx=2,
        operation='top_students',
        subject_name='Machine Learning',
        top_k=3,
        user_query='в машинном обучении покажи лучших студентов',
        expected_answer='Avery (80.4), John (76.9), Ethan (76.5)',
    ),
    TopStudentsCase(
        idx=3,
        operation='top_students',
        subject_name='Machine Learning',
        top_k=4,
        user_query='по машинному обучению топ-4 студентов',
        expected_answer='Avery (80.4), John (76.9), Ethan (76.5), Emma (75.8)',
    ),
    TopStudentsCase(
        idx=4,
        operation='top_students',
        subject_name='Probability Theory',
        top_k=3,
        user_query='теория вероятностей: кто лучшие студенты? топ3',
        expected_answer='Maria (79.1), Mia (74.9), Abigail (74.6)',
    ),
    TopStudentsCase(
        idx=5,
        operation='top_students',
        subject_name='Probability Theory',
        top_k=3,
        user_query='лучшие в теории вероятностей студенты',
        expected_answer='Maria (79.1), Mia (74.9), Abigail (74.6)',
    ),
    TopStudentsCase(
        idx=6,
        operation='top_students',
        subject_name='Probability Theory',
        top_k=10,
        user_query='топ 10 по теории вероятностей среди студентов',
        expected_answer='Maria (79.1), Mia (74.9), Abigail (74.6), Ella (73.8), David (73.4), Mila (72.6), Emily (72.4), Liam (71.4), Sofia (70.6), Ava (70.5)',
    ),
    TopStudentsCase(
        idx=7,
        operation='top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='теория оптимизации: кто из студентов лучший?',
        expected_answer='Harper (78.6), Evelyn (77.6), Mia (76.8)',
    ),
    TopStudentsCase(
        idx=8,
        operation='top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='среди студентов лучшие по теории оптимизации',
        expected_answer='Harper (78.6), Evelyn (77.6), Mia (76.8)',
    ),
    TopStudentsCase(
        idx=9,
        operation='top_students',
        subject_name='Optimization Theory',
        top_k=10,
        user_query='по теории оптимизации покажи студенческий топ-10',
        expected_answer='Harper (78.6), Evelyn (77.6), Mia (76.8), Alice (76.3), Avery (76.2), Charlotte (75.8), Bob (75.3), Ava (75.2), Elizabeth (74.6), Victoria (74.5)',
    ),
]

CASES_COMPLEX: list[TopStudentsCase] = [
    TopStudentsCase(
        idx=10,
        operation='top_students',
        subject_name='Machine Learning',
        top_k=3,
        user_query='топовые студенты по ml',
        expected_answer='Avery (80.4), John (76.9), Ethan (76.5)',
    ),
    TopStudentsCase(
        idx=11,
        operation='top_students',
        subject_name='Machine Learning',
        top_k=3,
        user_query='лучшие топ3 студенты по мл?',
        expected_answer='Avery (80.4), John (76.9), Ethan (76.5)',
    ),
    TopStudentsCase(
        idx=12,
        operation='top_students',
        subject_name='Machine Learning',
        top_k=2,
        user_query='топ2 студентов по машинке',
        expected_answer='Avery (80.4), John (76.9)',
    ),
    TopStudentsCase(
        idx=13,
        operation='top_students',
        subject_name='Machine Learning',
        top_k=7,
        user_query='лучшие по машинному обучению топ7',
        expected_answer='Avery (80.4), John (76.9), Ethan (76.5), Emma (75.8), Liam (75.5), Bob (74.9), Sofia (74.8)',
    ),
    TopStudentsCase(
        idx=14,
        operation='top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='три лучших студента по оптимизации',
        expected_answer='Harper (78.6), Evelyn (77.6), Mia (76.8)',
    ),
    TopStudentsCase(
        idx=15,
        operation='top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='топ-3 лучших студентов по оптам',
        expected_answer='Harper (78.6), Evelyn (77.6), Mia (76.8)',
    ),
    TopStudentsCase(
        idx=16,
        operation='top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='топовые студенты по метоптам?',
        expected_answer='Harper (78.6), Evelyn (77.6), Mia (76.8)',
    ),
    TopStudentsCase(
        idx=17,
        operation='top_students',
        subject_name='Machine Learning',
        top_k=3,
        user_query='топ учащихся по мл',
        expected_answer='Avery (80.4), John (76.9), Ethan (76.5)',
    ),
    TopStudentsCase(
        idx=18,
        operation='top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='топ студентов по методам оптимизации',
        expected_answer='Harper (78.6), Evelyn (77.6), Mia (76.8)',
    ),
    TopStudentsCase(
        idx=19,
        operation='top_students',
        subject_name='Probability Theory',
        top_k=4,
        user_query='топ4 лучших студентиков по теорверу',
        expected_answer='Maria (79.1), Mia (74.9), Abigail (74.6), Ella (73.8)',
    ),
    TopStudentsCase(
        idx=20,
        operation='top_students',
        subject_name='Probability Theory',
        top_k=3,
        user_query='лучшие топовые студенты по терверу?',
        expected_answer='Maria (79.1), Mia (74.9), Abigail (74.6)',
    ),
    TopStudentsCase(
        idx=21,
        operation='top_students',
        subject_name='Probability Theory',
        top_k=3,
        user_query='top студентов по теории вероятности',
        expected_answer='Maria (79.1), Mia (74.9), Abigail (74.6)',
    ),
    TopStudentsCase(
        idx=22,
        operation='top_students',
        subject_name='Machine Learning',
        top_k=5,
        user_query='топ пять учеников по мл',
        expected_answer='Avery (80.4), John (76.9), Ethan (76.5), Emma (75.8), Liam (75.5)',
    ),
    TopStudentsCase(
        idx=23,
        operation='top_students',
        subject_name='Probability Theory',
        top_k=8,
        user_query='топовые восемь студентов, предмет тервер',
        expected_answer='Maria (79.1), Mia (74.9), Abigail (74.6), Ella (73.8), David (73.4), Mila (72.6), Emily (72.4), Liam (71.4)',
    ),
    TopStudentsCase(
        idx=24,
        operation='top_students',
        subject_name='Machine Learning',
        top_k=2,
        user_query='по предмету мл топ два учащихся',
        expected_answer='Avery (80.4), John (76.9)',
    ),
]

_ENROLLMENTS_DF = pd.read_csv(ENROLLMENTS_CSV)
_GRADES_DF = pd.read_csv(META_GRADES_CSV)


def _format_top_students(rows: list[tuple[str, float]]) -> str:
    formatted = []
    for name, grade in rows:
        formatted.append(f'{name} ({grade:.1f})')
    return ', '.join(formatted)


def _expected_answer(subject_name: str, top_k: int) -> str:
    enrolled = _ENROLLMENTS_DF[_ENROLLMENTS_DF['subject_name'] == subject_name][
        'student_name'
    ]

    filtered = _GRADES_DF[
        (_GRADES_DF['subject_name'] == subject_name)
        & (_GRADES_DF['student_name'].isin(enrolled))
    ]
    if filtered.empty:
        return ''

    grouped = filtered.groupby('student_name')['grade'].mean().reset_index()
    grouped['grade'] = grouped['grade'].round(1)
    grouped = grouped.sort_values(by=['grade', 'student_name'], ascending=[False, True])
    top_rows = [
        (row.student_name, float(row.grade))
        for row in grouped.head(top_k).itertuples(index=False)
    ]
    return _format_top_students(top_rows)


def validate_cases(cases: list[TopStudentsCase], desc: str = 'validating top_students'):
    """Compute answers with pandas and print mismatches."""
    print(desc)
    ok = True
    for case in cases:
        actual_answer = _expected_answer(case.subject_name, case.top_k)
        if actual_answer != case.expected_answer:
            ok = False
            print('invalid case found')
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

    ok_simple = validate_cases(CASES_SIMPLE, 'validating top_students simple')
    if not ok_simple:
        print('ERROR: top_students simple cases are not valid')
        exit(1)
    print('...ok\n')

    ok_complex = validate_cases(CASES_COMPLEX, 'validating top_students complex')
    if not ok_complex:
        print('ERROR: top_students complex cases are not valid')
        exit(1)
    print('...ok\n')

    if args.print:
        print('\nCASES_SIMPLE :\n')
        for case in CASES_SIMPLE:
            print(f"        ('{case.user_query}',")
            print(f"         '{case.expected_answer}'),")
        print('\nCASES_COMPLEX :\n')
        for case in CASES_COMPLEX:
            print(f"        ('{case.user_query}',")
            print(f"         '{case.expected_answer}'),")
