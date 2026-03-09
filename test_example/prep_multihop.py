"""Run multihop prep cases and print expected results with mid actions."""

from __future__ import annotations

import argparse
from pprint import pformat
import re

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


def _collect_chunk_titles_with_substring(
    rag_result: dict[str, list[str]], needle: str
) -> list[str]:
    """Collect first lines of chunks containing substring."""
    needle_lower = needle.lower()
    matched_chunk_titles: list[str] = []
    for chunk in rag_result['chunks']:
        if needle_lower not in chunk.lower():
            continue
        first_row = chunk.strip().splitlines()[0].strip()
        matched_chunk_titles.append(first_row)
    return matched_chunk_titles


def _format_top_students(top_students: list[dict[str, float | str]]) -> str:
    """Format top students into readable one-line string."""
    return ', '.join(
        f'{row["name"]} ({float(row["score"]):.1f})' for row in top_students
    )


def _assert_matched_chunk_titles(matched_chunk_titles: list[str], min_count: int = 1):
    """Validate collected chunk titles count."""
    print(
        '\n(sanity check): len(matched_chunk_titles) = {} >= {} -> {}'.format(
            len(matched_chunk_titles), min_count, len(matched_chunk_titles) >= min_count
        )
    )
    assert len(matched_chunk_titles) >= min_count, (
        'len(matched_chunk_titles) should be >= {}'.format(min_count)
    )


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

    matched_chunk_titles = _collect_chunk_titles_with_substring(rag_result, lecturer)
    print(f'action_4: chunk titles with lecturer substring -> {matched_chunk_titles}')

    _assert_matched_chunk_titles(matched_chunk_titles, min_count=2)

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
    lecturer = COURSE_TO_LECTURER[simplest_subject]

    rag_query = f'кто лектор по курсу {simplest_subject}'
    rag_result = vector_search(query=rag_query, k=20)
    print(f'\naction_4: vector_search(query={rag_query!r}, k=...)')

    matched_chunk_titles = _collect_chunk_titles_with_substring(rag_result, lecturer)
    print(f'action_4: chunk titles with lecturer substring -> {matched_chunk_titles}')

    _assert_matched_chunk_titles(matched_chunk_titles, min_count=1)

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
    rag_result = vector_search(query=rag_query, k=20)

    # true values
    resolved_subject = 'Machine Learning'
    lecturer_short = 'Соколов Евгений'

    matched_chunk_titles = _collect_chunk_titles_with_substring(
        rag_result, lecturer_short
    )

    print(f'\naction_1: vector_search(query={rag_query!r}, k=...)')
    print(
        f'action_1: chunk titles with lecturer_short substring -> {matched_chunk_titles}'
    )

    top_students = get_top_students(resolved_subject)
    print(
        '\naction_2: '
        f'get_top_students(subject_name={resolved_subject!r}) -> {top_students}'
    )

    _assert_matched_chunk_titles(matched_chunk_titles, min_count=1)

    top_students_pretty = _format_top_students(top_students)
    expected_result = {
        'subject': resolved_subject,
        'top_students': top_students_pretty,
    }
    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


def run_case_5():
    """
    Multihop case 5: best students for course with seminarist Maksim Kaledin.
    """
    user_query = 'лучшие студенты по курсу при участии семинариста Каледина Максима'
    print(f'{ANSI_BOLD}{ANSI_CYAN}[5] Вопрос 5 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    # Каледин Максим is seminarist of Probability Theory AND lecturer of Random Processes
    seminarist = 'Каледин Максим'
    subject = 'Probability Theory'

    rag_query = 'по какому курсу семинарист Каледин Максим'
    rag_result = vector_search(query=rag_query, k=24)
    print(f'\naction_1: vector_search(query={rag_query!r}, k=...)')

    matched_chunk_titles = _collect_chunk_titles_with_substring(rag_result, seminarist)
    print(f'action_1: chunk titles with seminarist substring -> {matched_chunk_titles}')
    print(f'action_1: resolve subject from RAG chunks -> {subject}')

    top_students = get_top_students(subject)
    print(f'\naction_2: get_top_students(subject_name={subject!r}) -> {top_students}')

    _assert_matched_chunk_titles(matched_chunk_titles, min_count=2)

    top_students_pretty = _format_top_students(top_students)
    expected_result = {'subject': subject, 'top_students': top_students_pretty}
    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


def run_case_6():
    """
    Multihop case 6: surnames of top students for Tuesday lecture in P9.
    """
    user_query = (
        'фамилии топ студентов, которых можно застать во вторник на лекции в П9'
    )
    print(f'{ANSI_CYAN}[6] Вопрос 6 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    # true values
    resolved_subject = 'Optimization Theory'

    rag_query = 'лекция во вторник в аудитории П9'
    rag_result = vector_search(query=rag_query, k=20)
    print(f'\naction_1: vector_search(query={rag_query!r}, k=...)')

    matched_chunk_titles = _collect_chunk_titles_with_substring(rag_result, 'П9')
    print(f'action_1: chunk titles with location substring -> {matched_chunk_titles}')
    print(f'action_1: resolve subject from RAG chunks -> {resolved_subject}')

    top_students = get_top_students(resolved_subject)
    print(
        f'\naction_2: get_top_students(subject_name={resolved_subject!r}) -> {top_students}'
    )

    _assert_matched_chunk_titles(matched_chunk_titles, min_count=3)

    top_students_surnames = ', '.join(row['name'].split()[0] for row in top_students)
    expected_result = {
        'subject': resolved_subject,
        'top_student_surnames': top_students_surnames,
    }
    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


def run_case_7():
    """
    Multihop case 7: average score for subject with book author Semakov
    """
    user_query = 'средний балл по предмету, у которого автор учебника Семаков'
    print(f'{ANSI_CYAN}[7] Вопрос 7 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    # true values
    resolved_subject = 'Probability Theory'
    author_short = 'Семаков'

    rag_query = 'учебник Семаков'
    rag_result = vector_search(query=rag_query, k=20)
    print(f'\naction_1: vector_search(query={rag_query!r}, k=...)')

    matched_chunk_titles = _collect_chunk_titles_with_substring(
        rag_result, author_short
    )
    print(
        f'action_1: chunk titles with author_short substring -> {matched_chunk_titles}'
    )
    print(f'action_1: resolve subject from RAG chunks -> {resolved_subject}')

    avg_score = get_avg_score(resolved_subject)
    print(
        f'\naction_2: get_avg_score(subject_name={resolved_subject!r}) -> {avg_score}'
    )

    _assert_matched_chunk_titles(matched_chunk_titles, min_count=2)

    expected_result = {
        'subject': resolved_subject,
        'avg_score': avg_score['avg_score'],
    }
    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


def run_case_8():
    """
    Multihop case 8: books for subject with average equal to global average.
    """
    user_query = (
        'какие книги можно изучить студентам по направлению, '
        'средний балл по которому наиболее близок к общему среднему по предметам'
    )
    print(f'{ANSI_CYAN}[8] Вопрос 8 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    # true values
    resolved_subject = 'Методы оптимизации'
    expected_books = ['Numerical optimization', 'Convex optimization']

    subjects = sorted(students_df['subject_name'].dropna().unique().tolist())
    print(f'\naction_1: known subjects from pandas -> {subjects}\n')

    subject_to_avg: dict[str, float] = {}
    for subject in subjects:
        avg_result = get_avg_score(subject)
        print(f'action_1: get_avg_score(subject_name={subject!r}) -> {avg_result}')
        if avg_result:
            subject_to_avg[subject] = float(avg_result['avg_score'])

    overall_result = get_avg_overall_score()
    print(f'\naction_2: get_avg_overall_score() -> {overall_result}')
    overall_avg = float(overall_result['avg_score'])

    matched_subject = min(
        subject_to_avg.keys(),
        key=lambda item: (abs(subject_to_avg[item] - overall_avg), item),
    )
    matched_avg = subject_to_avg[matched_subject]
    print(
        '\naction_3: choose subject with avg_score closest to overall_avg -> '
        f'{matched_subject} ({matched_avg}), overall_avg={overall_avg}'
    )

    rag_query = f'литература по курсу {resolved_subject}'
    rag_result = vector_search(query=rag_query, k=20)
    print(f'\naction_4: rag_search(query={rag_query!r}, k=...)')

    matched_chunk_titles = _collect_chunk_titles_with_substring(
        rag_result, resolved_subject
    )
    print(f'action_4: chunk titles with subject substring -> {matched_chunk_titles}')
    print(f'\naction_5: extract literature bullet points -> {expected_books}')

    _assert_matched_chunk_titles(matched_chunk_titles, min_count=3)

    expected_result = {
        'subject': matched_subject,
        'subject_avg': matched_avg,
        'overall_avg': overall_avg,
        'literature': expected_books,
    }
    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


def run_case_9():
    """
    Multihop case 9: count students above ML passing score from related subjects.
    """
    user_query = (
        'курс по мл предполагает минимальный балл для успешного прохождения, '
        'сколько студентов из топ10 по смежным направлениям его превышают'
    )
    print(f'{ANSI_CYAN}[9] Вопрос 9 - {ANSI_BOLD}{user_query}{ANSI_RESET}')
    print(f'\nuser_query: {user_query}')

    # true values
    resolved_subject = 'Машинное обучение'
    passing_score = 4.3
    related_subjects = ['Probability Theory', 'Optimization Theory']

    rag_query = 'минимальный проходной балл курс машинного обучения'
    rag_result = vector_search(query=rag_query, k=20)
    print(f'\naction_1: rag_search(query={rag_query!r}, k=...)')

    matched_chunk_titles = _collect_chunk_titles_with_substring(
        rag_result, 'Минимальный проходной балл'
    )
    print(
        f'action_1: chunk titles with minimal passing score substring -> {matched_chunk_titles}'
    )
    print(f'action_1: resolve passing score from RAG chunks -> {passing_score}')

    _assert_matched_chunk_titles(matched_chunk_titles, min_count=2)

    related_top_students: list[dict[str, str | float]] = []
    for subject in related_subjects:
        top_students = get_top_students(subject, k=10)
        print(
            f'\naction_2: get_top_students(subject_name={subject!r}, k=10) -> {top_students}'
        )
        related_top_students.extend(top_students)

    count_above = sum(
        float(row['score']) > passing_score for row in related_top_students
    )
    print(
        '\naction_3: count scores > passing_score '
        f'({passing_score}) among related top10 -> {count_above}'
    )
    print(f'\n(sanity check): count_above > 0 -> {count_above > 0}')
    assert count_above > 0, 'count_above should be greater than 0'

    expected_result = {
        'subject': resolved_subject,
        'passing_score': passing_score,
        'count_above_passing_score': count_above,
        'related_subjects': related_subjects,
    }
    print(f'\n{ANSI_BOLD}expected_result:{ANSI_RESET} {expected_result}\n')
    print('-' * 80)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-1', dest='case_1', action='store_true')
    parser.add_argument('-2', dest='case_2', action='store_true')
    parser.add_argument('-3', dest='case_3', action='store_true')
    parser.add_argument('-4', dest='case_4', action='store_true')
    parser.add_argument('-5', dest='case_5', action='store_true')
    parser.add_argument('-6', dest='case_6', action='store_true')
    parser.add_argument('-7', dest='case_7', action='store_true')
    parser.add_argument('-8', dest='case_8', action='store_true')
    parser.add_argument('-9', dest='case_9', action='store_true')
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
    if args.case_6:
        selected_cases.append(6)
    if args.case_7:
        selected_cases.append(7)
    if args.case_8:
        selected_cases.append(8)
    if args.case_9:
        selected_cases.append(9)

    if not selected_cases:
        selected_cases = list(range(1, 10))

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
    if 7 in selected_cases:
        run_case_7()
    if 8 in selected_cases:
        run_case_8()
    if 9 in selected_cases:
        run_case_9()
