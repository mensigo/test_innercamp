"""Run books-oriented vector_search cases and print retrieved chunk previews."""

from __future__ import annotations

from dataclasses import dataclass

import argparse

from src.api import vector_search

TOP_K = 7
PREVIEW_LIMIT = 160
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_RESET = '\033[0m'


ML_BOOKS = """
Книги
- Hastie T., Tibshirani R, Friedman J. The Elements of Statistical Learning (2nd edition). Springer, 2009.
- Bishop C. M. Pattern Recognition and Machine Learning. Springer, 2006.
- Mohri M., Rostamizadeh A., Talwalkar A. Foundations of Machine Learning. MIT Press, 2012.
- Murphy K. Machine Learning: A Probabilistic Perspective. MIT Press, 2012.
- Mohammed J. Zaki, Wagner Meira Jr. Data Mining and Analysis. Fundamental Concepts and Algorithms. Cambridge University Press, 2014.
- Willi Richert, Luis Pedro Coelho. Building Machine Learning Systems with Python. Packt Publishing, 2013.
""".strip()

PROB_BOOKS = """
Литература

Гнеденко Б. В. Курс теории вероятностей. М.: Наука, 1988. [Прим. 1] 
Колмогоров А. Н. Основные понятия теории вероятностей. М.: Наука, 1974. [Прим. 1] 
Феллер В. Введение в теорию вероятностей и её приложения, в 2-х томах. М.: Мир, 1984. [Прим. 1] 
Боровков А. А. Теория вероятностей. М.: Наука, 1976. [Прим. 1] 
Розанов Ю. А. Теория вероятностей, случайные процессы и математическая статистика. М.: Наука, 1985. [Прим. 1] 
Прохоров Ю. В., Прохоров А. В. Курс лекций по теории вероятностей и математической статистике. М.: МЦНМО, 2019. 
Семаков С. Л. Элементы теории вероятностей и случайных процессов. М.: Физматлит, 2011. 
""".strip()

OPT_BOOKS = """
Рекомендуемая литература

1. J. Nocedal, S. Wright. Numerical optimization. Springer, 2006.
2. S. Boyd, L. Vandenberghe. Convex optimization. Cambridge University Press, 2004.
""".strip()


EXTRA_PHIL_BOOKS = """
Основная литература:
- Философия науки: учебник для магистратуры / под ред. А. И. Липкина. — 2-е изд., перераб. и доп. — М. : Издательство Юрайт, 2015.
- Семаков С.Л. Философия науки: история и методология. Учебное пособие. — М.: Дом интеллектуальной книги, 1998.
""".strip()

EXTRA_RAND_PROC_BOOKS = """
Литература
1. Коралов Л.Б., Синай Я.Г. Теория вероятностей и случайные процессы, МЦНМО, 2014.
2. Øksendal B. Stochastic Differential Equations: An Introduction with Applications, Springer, 2004, 10.1007/978-3-662-03185-8.
3. Ширяев А.Н. Основы стохастической финансовой математики (в двух томах), МЦНМО, 2016.
4. Булинский А.В., Ширяев А.Н. Теория случайных процессов, М: Физматлит, 2005.
5. Вентцель Е.С., Овчаров Л.А. Теория случайных процессов и её инженерные приложения, М:Высшая школа, 2000.
""".strip()

EXTRA_ML_BAYES_BOOKS = """
Литература

Barber D. Bayesian Reasoning and Machine Learning. Cambridge University Press, 2012.
Набор полезных фактов для матричных вычислений
Простые и удобные заметки по матричным вычислениям и свойствам гауссовских распределений
Памятка по теории вероятностей
Ветров Д.П., Кропотов Д.А. Байесовские методы машинного обучения, учебное пособие по спецкурсу, 2007 (Часть 1, PDF 1.22МБ; Часть 2, PDF 1.58МБ)
Bishop C.M. Pattern Recognition and Machine Learning. Springer, 2006.
Mackay D.J.C. Information Theory, Inference, and Learning Algorithms. Cambridge University Press, 2003.
Tipping M. Sparse Bayesian Learning. Journal of Machine Learning Research, 1, 2001, pp. 211-244.
Шумский С.А. Байесова регуляризация обучения. В сб. Лекции по нейроинформатике, часть 2, 2002.
""".strip()

EXTRA_ML_PRACTICE_BOOKS = """
Книги и статьи:

- Aurélien Géron. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
- Kevin P. Murphy. Probabilistic Machine Learning: An Introduction
- Charu C. Aggarwal. Recommender Systems: The Textbook
- Foster Provost, Tom Fawcett. Data Science for Business
""".strip()

EXTRA_ML_INTRO_BOOKS = """
Учебники

Онлайн-учебник по машинному обучению от ШАД.
Николенко С. Машинное обучение: основы, 2025. — 608 c.
Николенко С., Кадурин А., Архангельская Е. Глубокое обучение: основы, 2024. — 480 c.
Мэрфи К.П. Вероятностное машинное обучение. Введение, 2022. – 940 с.
Мэрфи К.П. Вероятностное машинное обучение. Дополнительные темы: основания, вывод, 2024. – 810 с.
Дайзенрот М. П, Альдо Фейзал А, Чен Сунь Он Питер. Математика в машинном обучении, 2024. – 512 с.
Уилке К. Основы визуализации данных: пособие по эффективной и убедительной подаче информации, 2024. – 352 с.
Шай Шалев-Шварц, Шай Бен-Давид. Идеи машинного обучения. От теории к алгоритмам, 2019. — 436 c.
Мерков А.Б. Распознавание образов. Введение в методы статистического обучения. 2011. 256 с.
Мерков А.Б. Распознавание образов. Построение и обучение вероятностных моделей. 2014. 238 с.
Коэльо Л.П., Ричарт В. Построение систем машинного обучения на языке Python. 2016. 302 с.
Hastie T., Tibshirani R., Friedman J. The Elements of Statistical Learning. Springer, 2014. — 739 p.
Bishop C.M. Pattern Recognition and Machine Learning. — Springer, 2006. — 738 p.
""".strip()

EXTRA_OPT_CONT_BOOKS = """
Книги

- Гасников А. В. Современные численные методы оптимизации. Метод универсального градиентного спуска. – М.: МФТИ, 2018.
- Гасников А. В. Презентации к избранным частям курса (наиболее важными являются презентации 1-4)
- Поляк Б.Т. Введение в оптимизацию. Изд. 2-ое, испр. и доп. – М.: ЛЕНАНД, 2014.
- Boyd S., Vandenberghe L. Convex optimization. – Cambridge University Press, 2004.
- Bubeck S. Convex optimization: algorithms and complexity. – Foundations and Trends in Machine Learning, 2015. – V. 8, N 3–4. – P. 231–357.
- Nemirovski A. Advanced Nonlinear Programming. – Lectures, ISyE 7683 Spring 2019.
- Nesterov Yu. Lectures on convex optimization. – Springer, 2018.
- Lan G. Lectures on optimization. Methods for Machine Learning, 2019.
""".strip()


@dataclass
class VectorSearchCase:
    idx: int
    user_query: str
    expected_answer: str


CASES_ML_BOOKS: list[VectorSearchCase] = [
    VectorSearchCase(
        idx=1,
        user_query='какие книги почитать по мл',
        expected_answer=ML_BOOKS,
    ),
    VectorSearchCase(
        idx=2,
        user_query='литература по машинному обучению',
        expected_answer=ML_BOOKS,
    ),
    VectorSearchCase(
        idx=3,
        user_query='книжки по машинке',
        expected_answer=ML_BOOKS,
    ),
    VectorSearchCase(
        idx=4,
        user_query='что можно почитать по мл',
        expected_answer=ML_BOOKS,
    ),
]

CASES_PROB_BOOKS: list[VectorSearchCase] = [
    VectorSearchCase(
        idx=5,
        user_query='литература по теории вероятности',
        expected_answer=PROB_BOOKS,
    ),
    VectorSearchCase(
        idx=6,
        user_query='по теорверу книги',
        expected_answer=PROB_BOOKS,
    ),
    VectorSearchCase(
        idx=7,
        user_query='книги для ознакомления с курсом теорвера',
        expected_answer=PROB_BOOKS,
    ),
    VectorSearchCase(
        idx=8,
        user_query='по теорверу рекомендуемая литература',
        expected_answer=PROB_BOOKS,
    ),
]

CASES_OPT_BOOKS: list[VectorSearchCase] = [
    VectorSearchCase(
        idx=9,
        user_query='что можно почитать по курсу оптимизации',
        expected_answer=OPT_BOOKS,
    ),
    VectorSearchCase(
        idx=10,
        user_query='книги по оптам',
        expected_answer=OPT_BOOKS,
    ),
    VectorSearchCase(
        idx=11,
        user_query='оптимизация список книжек',
        expected_answer=OPT_BOOKS,
    ),
    VectorSearchCase(
        idx=12,
        user_query='по оптимизации список рекомендованных произведений',
        expected_answer=OPT_BOOKS,
    ),
]

CASES_EXTRA_BOOKS: list[VectorSearchCase] = [
    VectorSearchCase(
        idx=13,
        user_query='книги по философии',
        expected_answer=EXTRA_PHIL_BOOKS,
    ),
    VectorSearchCase(
        idx=14,
        user_query='литература по случайным процессам',
        expected_answer=EXTRA_RAND_PROC_BOOKS,
    ),
    VectorSearchCase(
        idx=15,
        user_query='курс байесовской машинки, книги',
        expected_answer=EXTRA_ML_BAYES_BOOKS,
    ),
    VectorSearchCase(
        idx=16,
        user_query='что почитать по курсу мл на практике',
        expected_answer=EXTRA_ML_PRACTICE_BOOKS,
    ),
    VectorSearchCase(
        idx=17,
        user_query='учебники по курсу Воронцова',
        expected_answer=EXTRA_ML_INTRO_BOOKS,
    ),
    VectorSearchCase(
        idx=18,
        user_query='книги по курсу Гасникова',
        expected_answer=EXTRA_OPT_CONT_BOOKS,
    ),
]


def _first_row(text: str) -> str:
    """Get first non-empty row from chunk text."""
    for row in text.splitlines():
        stripped = row.strip()
        if stripped:
            return stripped
    return ''


def _shorten(text: str, limit: int = PREVIEW_LIMIT) -> str:
    """Shorten preview text for compact CLI output."""
    if len(text) <= limit:
        return text
    return f'{text[: limit - 3]}...'


def _find_expected_chunk_nums(chunks: list[str], expected_answer: str) -> list[int]:
    """Find all chunk numbers where expected_answer is a substring."""
    if not expected_answer:
        return []

    return [
        idx for idx, chunk in enumerate(chunks, start=1) if expected_answer in chunk
    ]


def _format_chunk_nums(chunk_nums: list[int]) -> str:
    """Format matched chunk numbers with color highlighting."""
    if not chunk_nums:
        return f'{COLOR_RED}None{COLOR_RESET}'

    first_num = f'{COLOR_GREEN}{chunk_nums[0]}{COLOR_RESET}'
    if len(chunk_nums) == 1:
        return first_num

    other_nums = ','.join(str(num) for num in chunk_nums[1:])
    return f'{first_num},{other_nums}'


def print_cases(
    cases: list[VectorSearchCase], title: str, show_full_chunk: bool = False
):
    """Run vector_search for each case and print chunk previews."""
    print(title)
    print(f'using top_k={TOP_K}\n')

    for case in cases:
        print(f'[{case.idx}] query: {case.user_query}')
        print('    ------------------------------------')
        result = vector_search(query=case.user_query, k=TOP_K)
        chunks = result['chunks']
        print(f'    retrieved_chunks: {len(chunks)}')
        expected_chunk_nums = _find_expected_chunk_nums(
            chunks=chunks, expected_answer=case.expected_answer
        )
        expected_chunk_nums_str = _format_chunk_nums(expected_chunk_nums)
        print(f'    expected_answer_chunk_nums: {expected_chunk_nums_str}')

        if not chunks:
            print('    - no chunks returned')
            print()
            continue

        for idx, chunk_text in enumerate(chunks, start=1):
            first_row = _shorten(_first_row(chunk_text))
            first_row = first_row or '<empty>'
            print(f'    - chunk_{idx} : {first_row}')
            if show_full_chunk:
                print(f'      chunk_{idx} full_text:\n{chunk_text}')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--print',
        action='store_true',
        help='print full chunk text in addition to first rows',
    )
    args = parser.parse_args()

    print_cases(
        CASES_ML_BOOKS,
        'vector_search ml (books)',
        show_full_chunk=args.print,
    )
    print_cases(
        CASES_PROB_BOOKS,
        'vector_search prob (books)',
        show_full_chunk=args.print,
    )
    print_cases(
        CASES_OPT_BOOKS,
        'vector_search opt (books)',
        show_full_chunk=args.print,
    )
    print_cases(
        CASES_EXTRA_BOOKS,
        'vector_search extra (books)',
        show_full_chunk=args.print,
    )
