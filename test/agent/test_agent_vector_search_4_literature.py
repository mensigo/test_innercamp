"""E2E checks for vector_search books cases through agent()."""

import pytest

from src.agent import agent

pytestmark = [pytest.mark.agent]

ML_BOOKS = """
The Elements of Statistical Learning
Pattern Recognition and Machine Learning
Foundations of Machine Learning
Machine Learning: A Probabilistic Perspective
Data Mining and Analysis. Fundamental Concepts and Algorithms
Building Machine Learning Systems with Python
""".strip().splitlines()

PROB_BOOKS = """
Курс теории вероятностей
Основные понятия теории вероятностей
Введение в теорию вероятностей и её приложения
Теория вероятностей
Теория вероятностей, случайные процессы и математическая статистика
Курс лекций по теории вероятностей и математической статистике
Элементы теории вероятностей и случайных процессов
""".strip().splitlines()

OPT_BOOKS = """
Numerical optimization
Convex optimization
""".strip().splitlines()


def _assert_contains_books(answer: str, expected_books: list[str]):
    answer_lower = answer.lower()
    for book in expected_books:
        assert book.lower() in answer_lower, f'missing book "{book}"'


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('какие книги почитать по мл', ML_BOOKS),
        ('литература по машинному обучению', ML_BOOKS),
        ('книжки по машинке', ML_BOOKS),
        ('что можно почитать по мл', ML_BOOKS),
    ],
)
def test_agent_e2e_vector_search_ml_books(query: str, expected_answer: list[str]):
    result = agent(query)
    answer = str(result.get('answer') or '')
    _assert_contains_books(answer, expected_answer)


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('литература по теории вероятности', PROB_BOOKS),
        ('по теорверу книги', PROB_BOOKS),
        ('книги для ознакомления с курсом теорвера', PROB_BOOKS),
        ('по теорверу рекомендуемая литература', PROB_BOOKS),
    ],
)
def test_agent_e2e_vector_search_prob_books(query: str, expected_answer: list[str]):
    result = agent(query)
    answer = str(result.get('answer') or '')
    _assert_contains_books(answer, expected_answer)


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('что можно почитать по курсу оптимизации', OPT_BOOKS),
        ('книги по оптам', OPT_BOOKS),
        ('оптимизация список книжек', OPT_BOOKS),
        ('по оптимизации список рекомендованных произведений', OPT_BOOKS),
    ],
)
def test_agent_e2e_vector_search_opt_books(query: str, expected_answer: list[str]):
    result = agent(query)
    answer = str(result.get('answer') or '')
    _assert_contains_books(answer, expected_answer)
