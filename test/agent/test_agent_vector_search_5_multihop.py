"""E2E checks for vector_search multihop cases through agent()."""

import pytest

from src.agent import agent

pytestmark = [pytest.mark.agent, pytest.mark.skip]


def test_agent_e2e_vector_search_multihop_hardest_course_lecturer():
    user_query = 'лектор самого сложного курса'
    expected_answer = 'Семаков Сергей Львович'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_vector_search_multihop_top10_opt_above_global_avg_count():
    user_query = 'сколько студентов из топ10 по оптам имеют балл выше общего среднего'
    expected_answer = '9'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_vector_search_multihop_top_students_by_sokolov_lecturer_course():
    user_query = 'топовые студенты по предмету лектора Евгения Соколова'
    expected_answer = 'Васильев Артем (4.9), Павлова Ирина (4.8)'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_vector_search_multihop_best_students_with_kaledin_seminarist():
    user_query = 'лучшие студенты по курсу при участии семинариста Каледина Максима'
    expected_answer = 'Козлова Татьяна (4.9), Петрушина Виктория (4.9)'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


def test_agent_e2e_vector_search_multihop_top_students_tuesday_p9_lecture_lastnames():
    user_query = (
        'фамилии топ студентов, которых можно застать во вторник на лекции в П9'
    )
    expected_answer = 'Ежик, Степанова'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.skip(reason='TODO: expected answer is not finalized yet')
def test_agent_e2e_vector_search_multihop_avg_grade_course_with_gnedenko_textbook_author():
    user_query = 'средний балл по предмету, у которого автор учебника Гнеденко'
    expected_answer = 'todo'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.skip(reason='TODO: expected answer is not finalized yet')
def test_agent_e2e_vector_search_multihop_best_main_course_lecturer_by_avg_student_grade():
    user_query = 'лучший из лекторов по основным курсам, если оценивать по среднему баллу студентов'
    expected_answer = 'todo'

    result = agent(user_query)
    assert str(result.get('answer') or '') == expected_answer
