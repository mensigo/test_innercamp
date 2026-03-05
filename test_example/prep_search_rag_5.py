"""Run multihop prep cases and print expected results with mid actions."""

from __future__ import annotations

import argparse
from pprint import pformat

import pandas as pd

from src.api import (
    get_avg_overall_score,
    get_avg_score,
    get_top_students,
    vector_search,
)

COURSE_TO_LECTURER = {
    'Machine Learning': 'Соколов Евгений Андреевич',
    'Probability Theory': 'Семаков Сергей Львович',
    'Optimization Theory': 'Кропотов Дмитрий Александрович',
}
ANSI_BOLD = '\033[1m'
ANSI_CYAN = '\033[96m'
ANSI_RESET = '\033[0m'

students_df = None


def run_case_1():
    """
    Multihop case 1: lecturer of the hardest core course.
    """
    user_query = 'кто лектор самого сложного курса'
    print(f'{ANSI_CYAN}[1] Вопрос 1 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    subjects = sorted(students_df['subject_name'].dropna().unique().tolist())
    print(f'\naction_1: known subjects from pandas -> {subjects}\n')

    subject_to_avg: dict[str, float] = {}
    for subject in subjects:
        avg_result = get_avg_score(subject)
        print(f'action_2: get_avg_score(subject_name={subject!r}) -> {avg_result}')
        if avg_result:
            subject_to_avg[subject] = float(avg_result['avg_score'])

    hardest_subject = min(
        subject_to_avg.keys(), key=lambda item: (subject_to_avg[item], item)
    )
    print(
        '\naction_3: choose hardest subject by minimal average score -> '
        f'{hardest_subject} ({subject_to_avg[hardest_subject]})'
    )

    # get answer globally
    lecturer = COURSE_TO_LECTURER[hardest_subject]

    rag_query = f'кто лектор по курсу {hardest_subject}'
    rag_result = vector_search(query=rag_query, k=20)
    print(f'\naction_4: vector_search(query={rag_query!r}, k=...)')

    matched_chunk_titles: list[str] = []
    for chunk in rag_result['chunks']:
        if lecturer.lower() not in chunk.lower():
            continue
        first_row = chunk.strip().splitlines()[0].strip()
        matched_chunk_titles.append(first_row)

    print(f'action_4: chunk titles with lecturer substring -> {matched_chunk_titles}')

    print(
        f'\n(sanity check): len(matched_chunk_titles) > 1 -> {len(matched_chunk_titles) > 0}'
    )
    assert len(matched_chunk_titles) > 1, (
        'len(matched_chunk_titles) should be greater than 1'
    )

    expected_result = {'subject': hardest_subject, 'lecturer': lecturer}
    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


def run_case_2():
    """
    Multihop case 2: lecturer of the simplest core course.
    """
    user_query = 'кто лектор самого простого курса'
    print(f'{ANSI_CYAN}[2] Вопрос 2 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    subjects = sorted(students_df['subject_name'].dropna().unique().tolist())
    print(f'\naction_1: known subjects from pandas -> {subjects}\n')

    subject_to_avg: dict[str, float] = {}
    for subject in subjects:
        avg_result = get_avg_score(subject)
        print(f'action_2: get_avg_score(subject_name={subject!r}) -> {avg_result}')
        if avg_result:
            subject_to_avg[subject] = float(avg_result['avg_score'])

    simplest_subject = max(
        subject_to_avg.keys(), key=lambda item: (subject_to_avg[item], item)
    )
    print(
        '\naction_3: choose simplest subject by maximal average score -> '
        f'{simplest_subject} ({subject_to_avg[simplest_subject]})'
    )

    # get answer globally
    lecturer = COURSE_TO_LECTURER.get(simplest_subject, 'NOT_FOUND')

    rag_query = f'кто лектор по курсу {simplest_subject}'
    rag_result = vector_search(query=rag_query, k=20)
    print(f'\naction_4: vector_search(query={rag_query!r}, k=...)')

    matched_chunk_titles: list[str] = []
    for chunk in rag_result['chunks']:
        if lecturer.lower() not in chunk.lower():
            continue
        first_row = chunk.strip().splitlines()[0].strip()
        matched_chunk_titles.append(first_row)

    print(f'action_4: chunk titles with lecturer substring -> {matched_chunk_titles}')

    expected_result = {'subject': simplest_subject, 'lecturer': lecturer}
    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


def run_case_3():
    """
    Multihop case 3: count top optimization students above global average.
    """
    user_query = 'сколько студентов из топ10 по оптам имеют балл выше общего среднего'
    print(f'{ANSI_CYAN}[3] Вопрос 3 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    overall_result = get_avg_overall_score()
    print(f'\naction_1: get_avg_overall_score() -> {overall_result}')
    overall_avg = float(overall_result['avg_score'])

    top_students = get_top_students('Optimization Theory', k=10)
    print(
        "\naction_2: get_top_students(subject_name='Optimization Theory', k=10) "
        f'-> {top_students}'
    )

    count_above = sum(1 for row in top_students if float(row['score']) > overall_avg)
    expected_result = {
        'overall_avg': overall_avg,
        'count_above_overall_avg': count_above,
    }
    print(
        '\naction_3: count scores > overall_avg '
        f'({overall_avg}) among top10 -> {count_above}'
    )
    print(f'\n(sanity check): count_above_overall_avg > 0 -> {count_above > 0}')
    assert count_above > 0, 'count_above_overall_avg should be greater than 0'

    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


def run_case_4():
    """
    Multihop case 4: top students for subject taught by lecturer Evgeniy Sokolov.
    """
    user_query = 'топовые студенты по предмету лектора Евгения Соколова'
    print(f'{ANSI_CYAN}[4] Вопрос 4 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    rag_query = 'какой предмет ведет лектор Евгений Соколов'
    rag_result = vector_search(query=rag_query, k=24)

    # true values
    resolved_subject = 'Machine Learning'
    lecturer_short = 'Соколов Евгений'

    matched_chunk_titles: list[str] = []
    for chunk in rag_result['chunks']:
        if lecturer_short.lower() not in chunk.lower():
            continue
        first_row = chunk.strip().splitlines()[0].strip()
        matched_chunk_titles.append(first_row)

    print(f'\naction_1: vector_search(query={rag_query!r}, k=...)')
    print(
        f'action_1: chunk titles with lecturer_short substring -> {matched_chunk_titles}'
    )

    top_students = get_top_students(resolved_subject)
    top_students_pretty = ', '.join(
        f'{row["name"]} ({float(row["score"]):.1f})' for row in top_students
    )
    print(
        '\naction_2: '
        f'get_top_students(subject_name={resolved_subject!r}) -> {top_students}'
    )

    expected_result = {
        'subject': resolved_subject,
        'top_students': top_students_pretty,
    }
    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


# WIP
def run_case_5():
    """
    Multihop case 5: best students for course with seminarist Maksim Kaledin.
    """
    user_query = 'лучшие студенты по курсу при участии семинариста Каледина Максима'
    print(f'{ANSI_BOLD}{ANSI_CYAN}[4] Вопрос 4 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    # Каледин Максим is seminarist of Probability Theory AND lecturer of Random Processes
    seminarist = 'Каледин Максим'
    lecturer = 'Семаков Сергей Львович'
    subject = 'Probability Theory'

    rag_query = 'по какому курсу семинарист Каледин Максим'
    rag_result = vector_search(query=rag_query, k=24)
    print(f'\naction_1: vector_search(query={rag_query!r}, k=...)')
    # TODO reorder seminarist name in Random Processes
    seminarist_lower = seminarist.lower()
    ll = []
    for chunk in rag_result['chunks']:
        if seminarist_lower in chunk.lower():
            first_line = chunk.strip().split('\n', 1)[0]
            ll.append(first_line)

    print(
        '\naction_1: chunk titles containing seminarist substring: {}'.format(
            ', '.join(ll)
        )
    )
    print(f'action_1: resolve subject from RAG chunks -> {subject}')

    top_students = get_top_students(subject)
    print(f'\naction_2: get_top_students(subject_name={subject!r}) -> {top_students}')

    expected_result = {'subject': subject, 'top_students': top_students}
    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


# WIP
def run_case_6():
    """Case 5: surnames of top students for Tuesday lecture in P9."""
    user_query = (
        'фамилии топ студентов, которых можно застать во вторник на лекции в П9'
    )
    print(f'{ANSI_CYAN}[5] Вопрос 5 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    rag_query = 'какие основные курсы читаются во вторник на лекции в П9'
    rag_result = vector_search(query=rag_query, k=7)
    print(
        f'\naction_1: vector_search(query={rag_query!r}, k=7) -> chunks={len(rag_result["chunks"])}'
    )

    resolved_subject = 'NOT_FOUND'
    for chunk in rag_result['chunks']:
        lower_chunk = chunk.lower()
        if (
            'вторник' in lower_chunk
            and 'п9' in lower_chunk
            and 'оптимизац' in lower_chunk
        ):
            resolved_subject = 'Optimization Theory'
            break
        if (
            'вторник' in lower_chunk
            and 'п9' in lower_chunk
            and 'машинное обучение' in lower_chunk
        ):
            resolved_subject = 'Machine Learning'
            break
        if (
            'вторник' in lower_chunk
            and 'п9' in lower_chunk
            and 'теория вероят' in lower_chunk
        ):
            resolved_subject = 'Probability Theory'
            break
    print(f'\naction_2: resolve subject from RAG chunks -> {resolved_subject}')

    top_students = []
    if resolved_subject != 'NOT_FOUND':
        top_students = get_top_students(resolved_subject, k=5)
    print(
        f'\naction_3: get_top_students(subject_name={resolved_subject!r}, k=5) -> {top_students}'
    )

    surnames: list[str] = []
    for row in top_students:
        name = str(row['name'])
        parts = name.split()
        if parts:
            surnames.append(parts[0])
    print(f'\naction_4: extract surnames from top students -> {surnames}')

    expected_result = {
        'subject': resolved_subject,
        'top_student_surnames': surnames,
    }
    print(f'expected_result: {expected_result}\n')
    print('-' * 80)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-1', dest='case_1', action='store_true')
    parser.add_argument('-2', dest='case_2', action='store_true')
    parser.add_argument('-3', dest='case_3', action='store_true')
    parser.add_argument('-4', dest='case_4', action='store_true')
    parser.add_argument('-5', dest='case_5', action='store_true')
    args = parser.parse_args()

    students_df = pd.read_csv('src/data/students.csv')

    print('multihop prep search rag #5')
    print('===========================')
    print(f'pandas_rows: {len(students_df)}')
    print(
        f'pandas_subjects: {sorted(students_df["subject_name"].dropna().unique().tolist())}\n'
    )
    print('===========================')
    print(f'COURSE_TO_LECTURER:\n{pformat(COURSE_TO_LECTURER)}\n\n')

    selected_cases = []
    if args.case_1:
        selected_cases.append(1)
    if args.case_2:
        selected_cases.append(2)
    if args.case_3:
        selected_cases.append(3)
    if args.case_4:
        selected_cases.append(4)
    if args.case_5:
        selected_cases.append(5)

    if not selected_cases:
        selected_cases = list(range(1, 7))

    if 1 in selected_cases:
        run_case_1()
    if 2 in selected_cases:
        run_case_2()
    if 3 in selected_cases:
        run_case_3()
    if 4 in selected_cases:
        run_case_4()
    if 5 in selected_cases:
        run_case_5()
    if 6 in selected_cases:
        run_case_6()
