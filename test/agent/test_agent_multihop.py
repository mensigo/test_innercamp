"""E2E checks for multihop cases through agent()."""

import pytest

from src.agent import agent

pytestmark = [pytest.mark.agent, pytest.mark.skip]


def test_agent_e2e_multihop_1_hardest_course_lecturer():
    user_query = 'лектор самого сложного курса'
    expected_answer = 'Семаков Сергей Львович'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_multihop_2_simplest_course_lecturer():
    user_query = 'лектор самого простого курса'
    expected_answer = 'Соколов Евгений Андреевич'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_multihop_3_top10_opt_above_global_avg_count():
    user_query = 'сколько студентов из топ10 по оптам имеют балл выше общего среднего'
    expected_answer = '9'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_multihop_4_top_students_with_specified_lecturer():
    user_query = 'топовые студенты по предмету лектора Евгения Соколова'
    expected_answer = 'Васильев Артем (4.9), Павлова Ирина (4.8)'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_multihop_5_best_students_with_specified_seminarist():
    user_query = 'лучшие студенты по курсу при участии семинариста Каледина Максима'
    expected_answer = 'Козлова Татьяна (4.9), Петрушина Виктория (4.9)'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_multihop_6_top_students_lastnames_with_specified_lecture_schedule():
    user_query = (
        'фамилии топ студентов, которых можно застать во вторник на лекции в П9'
    )
    expected_answer = 'Ежик, Степанова'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_multihop_7_avg_grade_for_course_with_specified_textbook_author():
    user_query = 'средний балл по предмету, у которого автор учебника Семаков'
    expected_answer = '3.9'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_multihop_8_literature_for_course_closest_to_global_avg():
    user_query = (
        'какие книги можно изучить студентам по направлению, '
        'средний балл по которому наиболее близок к общему среднему по предметам'
    )
    expected_books = ['Numerical optimization', 'Convex optimization']

    result = agent(user_query)
    answer = str(result.get('answer') or '').lower()
    for book in expected_books:
        assert book.lower() in answer, f'expected to find {book!r} in answer'


def test_agent_e2e_multihop_9_students_above_ml_passing_score():
    user_query = (
        'курс по мл предполагает минимальный балл для успешного прохождения, '
        'сколько студентов из топ10 по смежным направлениям его превышают'
    )
    expected_answer = '12'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer
