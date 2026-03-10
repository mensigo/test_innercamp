"""Prep hardcoded hardest_questions e2e data and validate against CSVs."""

from __future__ import annotations

from dataclasses import dataclass

import argparse
import json
import pandas as pd

from src_example.data_store import META_GRADES_CSV


@dataclass
class HardQuestionsCase:
    idx: int
    operation: str
    subject_name: str
    top_k: int
    user_query: str
    expected_answer: str


CASES: list[HardQuestionsCase] = [
    HardQuestionsCase(
        idx=1,
        operation='hardest_questions',
        subject_name='Machine Learning',
        top_k=3,
        user_query='самые сложные вопросы по машинному обучению',
        expected_answer='[{"question_id": 7, "avg_grade": 69.1}, {"question_id": 9, "avg_grade": 71.2}, {"question_id": 10, "avg_grade": 71.9}]',
    ),
    HardQuestionsCase(
        idx=2,
        operation='hardest_questions',
        subject_name='Machine Learning',
        top_k=3,
        user_query='топ-3 самых трудных вопросов по машинке',
        expected_answer='[{"question_id": 7, "avg_grade": 69.1}, {"question_id": 9, "avg_grade": 71.2}, {"question_id": 10, "avg_grade": 71.9}]',
    ),
    HardQuestionsCase(
        idx=3,
        operation='hardest_questions',
        subject_name='Probability Theory',
        top_k=3,
        user_query='топ-3 сложных вопросов по теории вероятностей',
        expected_answer='[{"question_id": 4, "avg_grade": 36.8}, {"question_id": 7, "avg_grade": 70.1}, {"question_id": 8, "avg_grade": 70.7}]',
    ),
    HardQuestionsCase(
        idx=4,
        operation='hardest_questions',
        subject_name='Probability Theory',
        top_k=3,
        user_query='какие самые сложные вопросы в тервере?',
        expected_answer='[{"question_id": 4, "avg_grade": 36.8}, {"question_id": 7, "avg_grade": 70.1}, {"question_id": 8, "avg_grade": 70.7}]',
    ),
    HardQuestionsCase(
        idx=5,
        operation='hardest_questions',
        subject_name='Optimization Theory',
        top_k=3,
        user_query='в теории оптимизации покажи топ-3 сложных вопросов',
        expected_answer='[{"question_id": 9, "avg_grade": 67.4}, {"question_id": 5, "avg_grade": 68.7}, {"question_id": 2, "avg_grade": 70.4}]',
    ),
    HardQuestionsCase(
        idx=6,
        operation='hardest_questions',
        subject_name='Machine Learning',
        top_k=5,
        user_query='дай топ-5 самых сложных вопросов по мл',
        expected_answer='[{"question_id": 7, "avg_grade": 69.1}, {"question_id": 9, "avg_grade": 71.2}, {"question_id": 10, "avg_grade": 71.9}, {"question_id": 6, "avg_grade": 72.0}, {"question_id": 4, "avg_grade": 72.4}]',
    ),
    HardQuestionsCase(
        idx=7,
        operation='hardest_questions',
        subject_name='Probability Theory',
        top_k=5,
        user_query='покажи 5 сложнейших задач по теории вероятности',
        expected_answer='[{"question_id": 4, "avg_grade": 36.8}, {"question_id": 7, "avg_grade": 70.1}, {"question_id": 8, "avg_grade": 70.7}, {"question_id": 10, "avg_grade": 71.5}, {"question_id": 5, "avg_grade": 72.3}]',
    ),
    HardQuestionsCase(
        idx=8,
        operation='hardest_questions',
        subject_name='Optimization Theory',
        top_k=5,
        user_query='в оптимизации топ-5 трудных вопросов',
        expected_answer='[{"question_id": 9, "avg_grade": 67.4}, {"question_id": 5, "avg_grade": 68.7}, {"question_id": 2, "avg_grade": 70.4}, {"question_id": 6, "avg_grade": 70.9}, {"question_id": 8, "avg_grade": 72.2}]',
    ),
    HardQuestionsCase(
        idx=9,
        operation='hardest_questions',
        subject_name='Machine Learning',
        top_k=3,
        user_query='самые трудные вопросы по machine learning? топ3',
        expected_answer='[{"question_id": 7, "avg_grade": 69.1}, {"question_id": 9, "avg_grade": 71.2}, {"question_id": 10, "avg_grade": 71.9}]',
    ),
    HardQuestionsCase(
        idx=10,
        operation='hardest_questions',
        subject_name='Probability Theory',
        top_k=3,
        user_query='в тервере самые трудные вопросы топ три',
        expected_answer='[{"question_id": 4, "avg_grade": 36.8}, {"question_id": 7, "avg_grade": 70.1}, {"question_id": 8, "avg_grade": 70.7}]',
    ),
]

_GRADES_DF = pd.read_csv(META_GRADES_CSV)


def _format_hardest(rows: list[dict]) -> str:
    return json.dumps(rows, ensure_ascii=False)


def _expected_answer(subject_name: str, top_k: int) -> str:
    filtered = _GRADES_DF[_GRADES_DF['subject_name'] == subject_name]
    if filtered.empty:
        return '[]'

    grouped = (
        filtered.groupby('question_id')['grade']
        .mean()
        .reset_index()
        .rename(columns={'grade': 'avg_grade'})
    )
    grouped['avg_grade'] = grouped['avg_grade'].round(1)
    grouped = grouped.sort_values(
        by=['avg_grade', 'question_id'], ascending=[True, True]
    )

    rows = [
        {'question_id': int(row.question_id), 'avg_grade': float(row.avg_grade)}
        for row in grouped.head(top_k).itertuples(index=False)
    ]
    return _format_hardest(rows)


def validate_cases(
    cases: list[HardQuestionsCase], desc: str = 'validating hardest_questions'
):
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

    ok = validate_cases(CASES, 'validating hardest_questions cases')
    if not ok:
        print('ERROR: hardest_questions cases are not valid')
        exit(1)
    print('...ok\n')

    if args.print:
        print('\nCASES:\n')
        for case in CASES:
            print(f"        ('{case.user_query}',")
            print(f"         '{case.expected_answer}'),")
