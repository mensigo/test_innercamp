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


EXTRA_PHIL_BOOKS = """
Философия науки: учебник для магистратуры
Философия науки: история и методология
""".strip().splitlines()

EXTRA_RAND_PROC_BOOKS = """
Теория вероятностей и случайные процессы
Stochastic Differential Equations: An Introduction with Applications
Основы стохастической финансовой математики
Теория случайных процессов
""".strip().splitlines()

EXTRA_ML_BAYES_BOOKS = """
Bayesian Reasoning and Machine Learning
Байесовские методы машинного обучения
Pattern Recognition and Machine Learning
Information Theory, Inference, and Learning Algorithms
Sparse Bayesian Learning
Байесова регуляризация обучения
""".strip().splitlines()

EXTRA_ML_PRACTICE_BOOKS = """
Hands-On Machine Learning
Probabilistic Machine Learning
Recommender Systems
Data Science for Business
""".strip().splitlines()

EXTRA_ML_INTRO_BOOKS = """
Онлайн-учебник по машинному обучению от ШАД
Глубокое обучение
Вероятностное машинное обучение
Математика в машинном обучении
Основы визуализации данных
Идеи машинного обучения
Распознавание образов
The Elements of Statistical Learning
Pattern Recognition and Machine Learning
""".strip().splitlines()

EXTRA_OPT_CONT_BOOKS = """
Современные численные методы оптимизации
Введение в оптимизацию
Convex optimization
Convex optimization: algorithms and complexity
Advanced Nonlinear Programming
Lectures on convex optimization
Lectures on optimization. Methods for Machine Learning
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


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('книги по философии', EXTRA_PHIL_BOOKS),
        ('литература по случайным процессам', EXTRA_RAND_PROC_BOOKS),
        ('курс байесовской машинки, книги', EXTRA_ML_BAYES_BOOKS),
        ('что почитать по курсу мл на практике', EXTRA_ML_PRACTICE_BOOKS),
        ('учебники по курсу Воронцова', EXTRA_ML_INTRO_BOOKS),
        ('книги по курсу Гасникова', EXTRA_OPT_CONT_BOOKS),
    ],
)
def test_agent_e2e_vector_search_extra_books(query: str, expected_answer: list[str]):
    result = agent(query)
    answer = str(result.get('answer') or '')
    _assert_contains_books(answer, expected_answer)
