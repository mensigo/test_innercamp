"""Prep hardcoded get_top_students e2e data and validate against students.csv."""

from __future__ import annotations

from dataclasses import dataclass

import argparse
import pandas as pd

from src.prepare_data import ensure_students_csv


@dataclass
class TopStudentsCase:
    idx: int
    tool_name: str
    subject_name: str
    top_k: int
    user_query: str
    expected_answer: str


CASES_SIMPLE: list[TopStudentsCase] = [
    TopStudentsCase(
        idx=1,
        tool_name='get_top_students',
        subject_name='Machine Learning',
        top_k=3,
        user_query='машинное обучение: лучшие студенты',
        expected_answer='Васильев Артем (4.9), Павлова Ирина (4.8), Новикова Дарья (4.7)',
    ),
    TopStudentsCase(
        idx=2,
        tool_name='get_top_students',
        subject_name='Machine Learning',
        top_k=3,
        user_query='в машинном обучении покажи лучших студентов',
        expected_answer='Васильев Артем (4.9), Павлова Ирина (4.8), Новикова Дарья (4.7)',
    ),
    TopStudentsCase(
        idx=3,
        tool_name='get_top_students',
        subject_name='Machine Learning',
        top_k=4,
        user_query='по машинному обучению топ-4 студентов',
        expected_answer='Васильев Артем (4.9), Павлова Ирина (4.8), Новикова Дарья (4.7), Козлов Кирилл (4.6)',
    ),
    TopStudentsCase(
        idx=4,
        tool_name='get_top_students',
        subject_name='Probability Theory',
        top_k=3,
        user_query='теория вероятностей: кто лучшие студенты? топ3',
        expected_answer='Козлова Татьяна (4.9), Петрушина Виктория (4.9), Никитина Наталья (4.8)',
    ),
    TopStudentsCase(
        idx=5,
        tool_name='get_top_students',
        subject_name='Probability Theory',
        top_k=3,
        user_query='лучшие в теории вероятностей студенты',
        expected_answer='Козлова Татьяна (4.9), Петрушина Виктория (4.9), Никитина Наталья (4.8)',
    ),
    TopStudentsCase(
        idx=6,
        tool_name='get_top_students',
        subject_name='Probability Theory',
        top_k=10,
        user_query='топ 10 по теории вероятностей среди студентов',
        expected_answer=(
            'Козлова Татьяна (4.9), Петрушина Виктория (4.9), Никитина Наталья (4.8), '
            'Новикова Дарья (4.8), Иванова Анастасия (4.6), Орлова Мария (4.2), '
            'Степанов Тимофей (4.2), Ежик Ольга (4.1), Макаров Михаил (4.1), '
            'Степанова Ксения (4.1)'
        ),
    ),
    TopStudentsCase(
        idx=7,
        tool_name='get_top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='теория оптимизации: кто из студентов лучший?',
        expected_answer='Ежик Ольга (5.0), Степанова Ксения (5.0), Андреев Матвей (4.9)',
    ),
    TopStudentsCase(
        idx=8,
        tool_name='get_top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='среди студентов лучшие по теории оптимизации',
        expected_answer='Ежик Ольга (5.0), Степанова Ксения (5.0), Андреев Матвей (4.9)',
    ),
    TopStudentsCase(
        idx=9,
        tool_name='get_top_students',
        subject_name='Optimization Theory',
        top_k=10,
        user_query='по теории оптимизации покажи студенческий топ-10',
        expected_answer=(
            'Ежик Ольга (5.0), Степанова Ксения (5.0), Андреев Матвей (4.9), '
            'Смирнов Александр (4.8), Николаева Юлия (4.7), Орлова Мария (4.6), '
            'Васильев Артем (4.4), Лебедев Иван (4.2), Козлов Кирилл (4.1), '
            'Петрушина Виктория (4.0)'
        ),
    ),
]

CASES_COMPLEX: list[TopStudentsCase] = [
    TopStudentsCase(
        idx=10,
        tool_name='get_top_students',
        subject_name='Machine Learning',
        top_k=3,
        user_query='топовые студенты по ml',
        expected_answer='Васильев Артем (4.9), Павлова Ирина (4.8), Новикова Дарья (4.7)',
    ),
    TopStudentsCase(
        idx=11,
        tool_name='get_top_students',
        subject_name='Machine Learning',
        top_k=3,
        user_query='лучшие топ3 студенты по мл?',
        expected_answer='Васильев Артем (4.9), Павлова Ирина (4.8), Новикова Дарья (4.7)',
    ),
    TopStudentsCase(
        idx=12,
        tool_name='get_top_students',
        subject_name='Machine Learning',
        top_k=2,
        user_query='топ2 студентов по машинке',
        expected_answer='Васильев Артем (4.9), Павлова Ирина (4.8)',
    ),
    TopStudentsCase(
        idx=13,
        tool_name='get_top_students',
        subject_name='Machine Learning',
        top_k=7,
        user_query='лучшие по машинному обучению топ7',
        expected_answer='Васильев Артем (4.9), Павлова Ирина (4.8), Новикова Дарья (4.7), Козлов Кирилл (4.6), Волкова Екатерина (4.5), Козлова Татьяна (4.5), Николаева Юлия (4.5)',
    ),
    TopStudentsCase(
        idx=14,
        tool_name='get_top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='три лучших студента по оптимизации',
        expected_answer='Ежик Ольга (5.0), Степанова Ксения (5.0), Андреев Матвей (4.9)',
    ),
    TopStudentsCase(
        idx=15,
        tool_name='get_top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='топ-3 лучших студентов по оптам',
        expected_answer='Ежик Ольга (5.0), Степанова Ксения (5.0), Андреев Матвей (4.9)',
    ),
    TopStudentsCase(
        idx=16,
        tool_name='get_top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='топовые студенты по метоптам?',
        expected_answer='Ежик Ольга (5.0), Степанова Ксения (5.0), Андреев Матвей (4.9)',
    ),
    TopStudentsCase(
        idx=17,
        tool_name='get_top_students',
        subject_name='Machine Learning',
        top_k=3,
        user_query='топ учащихся по мл',
        expected_answer='Васильев Артем (4.9), Павлова Ирина (4.8), Новикова Дарья (4.7)',
    ),
    TopStudentsCase(
        idx=18,
        tool_name='get_top_students',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='топ студентов по методам оптимизации',
        expected_answer='Ежик Ольга (5.0), Степанова Ксения (5.0), Андреев Матвей (4.9)',
    ),
    TopStudentsCase(
        idx=19,
        tool_name='get_top_students',
        subject_name='Probability Theory',
        top_k=4,
        user_query='топ4 лучших студентиков по теорверу',
        expected_answer='Козлова Татьяна (4.9), Петрушина Виктория (4.9), Никитина Наталья (4.8), Новикова Дарья (4.8)',
    ),
    TopStudentsCase(
        idx=20,
        tool_name='get_top_students',
        subject_name='Probability Theory',
        top_k=3,
        user_query='лучшие топовые студенты по терверу?',
        expected_answer='Козлова Татьяна (4.9), Петрушина Виктория (4.9), Никитина Наталья (4.8)',
    ),
    TopStudentsCase(
        idx=21,
        tool_name='get_top_students',
        subject_name='Probability Theory',
        top_k=3,
        user_query='top студентов по теории вероятности',
        expected_answer='Козлова Татьяна (4.9), Петрушина Виктория (4.9), Никитина Наталья (4.8)',
    ),
    TopStudentsCase(
        idx=22,
        tool_name='get_top_students',
        subject_name='Machine Learning',
        top_k=5,
        user_query='топ пять учеников по мл',
        expected_answer='Васильев Артем (4.9), Павлова Ирина (4.8), Новикова Дарья (4.7), Козлов Кирилл (4.6), Волкова Екатерина (4.5)',
    ),
    TopStudentsCase(
        idx=23,
        tool_name='get_top_students',
        subject_name='Probability Theory',
        top_k=8,
        user_query='топовые восемь студентов, предмет тервер',
        expected_answer='Козлова Татьяна (4.9), Петрушина Виктория (4.9), Никитина Наталья (4.8), Новикова Дарья (4.8), Иванова Анастасия (4.6), Орлова Мария (4.2), Степанов Тимофей (4.2), Ежик Ольга (4.1)',
    ),
    TopStudentsCase(
        idx=24,
        tool_name='get_top_students',
        subject_name='Machine Learning',
        top_k=2,
        user_query='по предмету мл топ два учащихся',
        expected_answer='Васильев Артем (4.9), Павлова Ирина (4.8)',
    ),
]

_STUDENTS_CSV_PATH = ensure_students_csv(force=False)
_STUDENTS_DF = pd.read_csv(_STUDENTS_CSV_PATH)


def _format_top_students(rows: list[tuple[str, float]]) -> str:
    formatted = []
    for name, score in rows:
        formatted.append(f'{name} ({score:.1f})')
    return ', '.join(formatted)


def _expected_answer(subject_name: str, top_k: int) -> str:
    filtered = _STUDENTS_DF[_STUDENTS_DF['subject_name'] == subject_name].copy()
    if filtered.empty:
        return ''

    filtered['score'] = filtered['score'].astype(float).round(1)
    filtered = filtered.sort_values(
        by=['score', 'student_name'], ascending=[False, True]
    )
    top_rows = [
        (str(row.student_name), float(row.score))
        for row in filtered.head(top_k).itertuples(index=False)
    ]
    return _format_top_students(top_rows)


def validate_cases(
    cases: list[TopStudentsCase], desc: str = 'validating get_top_students'
):
    """Validate case metadata and expected answers against generated dataset."""
    print(desc)
    ok = True
    known_subjects = set(_STUDENTS_DF['subject_name'])
    for case in cases:
        if case.tool_name != 'get_top_students':
            ok = False
            print(f'invalid tool_name for idx={case.idx}: {case.tool_name}')
        if case.top_k <= 0:
            ok = False
            print(f'invalid top_k for idx={case.idx}: {case.top_k}')
        if case.subject_name not in known_subjects:
            ok = False
            print(f'unknown subject for idx={case.idx}: {case.subject_name}')

        actual_answer = _expected_answer(case.subject_name, case.top_k)
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

    ok_simple = validate_cases(CASES_SIMPLE, 'validating get_top_students simple')
    if not ok_simple:
        print('ERROR: get_top_students simple cases are not valid')
        exit(1)
    print('...ok\n')

    ok_complex = validate_cases(CASES_COMPLEX, 'validating get_top_students complex')
    if not ok_complex:
        print('ERROR: get_top_students complex cases are not valid')
        exit(1)
    print('...ok\n')

    if args.print:
        print('\nCASES_SIMPLE:\n')
        for case in CASES_SIMPLE:
            computed = _expected_answer(case.subject_name, case.top_k)
            print(f"\t\t('{case.user_query}',")
            print(f"\t\t'{case.expected_answer}')")
        print('\nCASES_COMPLEX:\n')
        for case in CASES_COMPLEX:
            computed = _expected_answer(case.subject_name, case.top_k)
            print(f"\t\t('{case.user_query}',")
            print(f"\t\t'{case.expected_answer}')")
