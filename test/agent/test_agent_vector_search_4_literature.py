"""E2E checks for vector_search books cases through agent()."""

import pytest

from src.agent import agent

pytestmark = [pytest.mark.agent, pytest.mark.skip]

ML_BOOKS = """
- Hastie T., Tibshirani R, Friedman J. The Elements of Statistical Learning (2nd edition). Springer, 2009.
- Bishop C. M. Pattern Recognition and Machine Learning. Springer, 2006.
- Mohri M., Rostamizadeh A., Talwalkar A. Foundations of Machine Learning. MIT Press, 2012.
- Murphy K. Machine Learning: A Probabilistic Perspective. MIT Press, 2012.
- Mohammed J. Zaki, Wagner Meira Jr. Data Mining and Analysis. Fundamental Concepts and Algorithms. Cambridge University Press, 2014.
- Willi Richert, Luis Pedro Coelho. Building Machine Learning Systems with Python. Packt Publishing, 2013.
""".strip()

PROB_BOOKS = """
- Гнеденко Б. В. Курс теории вероятностей. М.: Наука, 1988.
- Колмогоров А. Н. Основные понятия теории вероятностей. М.: Наука, 1974.
- Феллер В. Введение в теорию вероятностей и её приложения, в 2-х томах. М.: Мир, 1984.
- Боровков А. А. Теория вероятностей. М.: Наука, 1976.
- Розанов Ю. А. Теория вероятностей, случайные процессы и математическая статистика. М.: Наука, 1985.
- Прохоров Ю. В., Прохоров А. В. Курс лекций по теории вероятностей и математической статистике. М.: МЦНМО, 2019.
- Семаков С. Л. Элементы теории вероятностей и случайных процессов. М.: Физматлит, 2011.
""".strip()

OPT_BOOKS = """
- J. Nocedal, S. Wright. Numerical optimization. Springer, 2006.
- S. Boyd, L. Vandenberghe. Convex optimization. Cambridge University Press, 2004.
""".strip()


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('какие книги почитать по мл', ML_BOOKS),
        ('литература по машинному обучению', ML_BOOKS),
        ('книжки по машинке', ML_BOOKS),
        ('что можно почитать по мл', ML_BOOKS),
    ],
)
def test_agent_e2e_vector_search_ml_books(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('литература по теории вероятности', PROB_BOOKS),
        ('по теорверу книги', PROB_BOOKS),
        ('книги для ознакомления с курсом теорвера', PROB_BOOKS),
        ('по теорверу рекомендуемая литература', PROB_BOOKS),
    ],
)
def test_agent_e2e_vector_search_prob_books(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer


@pytest.mark.parametrize(
    ('query', 'expected_answer'),
    [
        ('что можно почитать по курсу оптимизации', OPT_BOOKS),
        ('книги по оптам', OPT_BOOKS),
        ('оптимизация список книжек', OPT_BOOKS),
        ('по оптимизации список рекомендованных произведений', OPT_BOOKS),
    ],
)
def test_agent_e2e_vector_search_opt_books(query: str, expected_answer: str):
    result = agent(query)
    assert str(result.get('answer') or '') == expected_answer
