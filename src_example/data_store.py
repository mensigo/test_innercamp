"""Generate and load deterministic local data files for src_learn agent."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from src_learn.logger import logger

SUBJECTS = [
    'Machine Learning',
    'Probability Theory',
    'Optimization Theory',
]

STUDENTS = [
    'Abigail',
    'Alice',
    'Amelia',
    'Aria',
    'Ava',
    'Avery',
    'Bob',
    'Camila',
    'Charlotte',
    'David',
    'Elizabeth',
    'Ella',
    'Emily',
    'Emma',
    'Ethan',
    'Evelyn',
    'Harper',
    'Isabella',
    'John',
    'Liam',
    'Madison',
    'Maria',
    'Mia',
    'Mila',
    'Noah',
    'Olivia',
    'Scarlett',
    'Sofia',
    'Sophia',
    'Victoria',
]

DATA_ROOT = Path(__file__).parent / 'generated_data'
EXAMS_DIR = DATA_ROOT / 'exams'
STUDENTS_CSV = DATA_ROOT / 'students.csv'
ENROLLMENTS_CSV = DATA_ROOT / 'enrollments.csv'
META_GRADES_CSV = DATA_ROOT / 'meta_grades.csv'


def _build_exam_markdown(subject_name: str) -> str:
    """Build markdown file with 10 deterministic question-answer pairs."""
    if subject_name == 'Machine Learning':
        qa = [
            (
                'Назовите 4 ключевые особенности машинного обучения.',
                '- обобщающая способность\n- обучение по данным\n- представление признаков\n- оптимизация',
            ),
            (
                'Что такое переобучение?',
                'Переобучение возникает, когда модель запоминает шум обучающей выборки и плохо работает на новых данных.',
            ),
            (
                'Что такое недообучение?',
                'Недообучение возникает, когда модель слишком проста и не улавливает закономерности в данных.',
            ),
            (
                'Что такое регуляризация?',
                'Регуляризация добавляет ограничения или штрафы, чтобы уменьшить риск переобучения.',
            ),
            (
                'Что такое градиентный спуск?',
                'Градиентный спуск итеративно обновляет параметры в направлении уменьшения функции потерь.',
            ),
            (
                'Что такое кросс-валидация?',
                'Кросс-валидация делит данные на фолды для оценки обобщающей способности модели.',
            ),
            (
                'Назовите распространенные метрики качества классификации.',
                'К распространенным метрикам классификации относятся accuracy, precision, recall, F1-score и ROC-AUC.',
            ),
            (
                'Что такое масштабирование признаков?',
                'Масштабирование признаков приводит признаки к сопоставимым диапазонам для более стабильной оптимизации.',
            ),
            (
                'Что такое компромисс смещения и дисперсии?',
                'Это баланс между простотой модели (смещение) и чувствительностью к данным (дисперсия).',
            ),
            (
                'Какой параметр отличает Adam от AdamW?',
                'В AdamW штраф weight decay отделен от градиентного обновления, в отличие от классической L2-реализации в Adam.',
            ),
        ]
    elif subject_name == 'Probability Theory':
        qa = [
            (
                'Что такое случайная величина?',
                'Случайная величина отображает исходы случайного эксперимента в числовые значения.',
            ),
            (
                'Назовите 3 свойства случайной величины.',
                '- измеримость\n- наличие вероятностного распределения\n- возможность определить математическое ожидание и дисперсию',
            ),
            (
                'Что такое математическое ожидание?',
                'Математическое ожидание — это среднее значение случайной величины, взвешенное по вероятностям.',
            ),
            (
                'Что такое дисперсия?',
                'Дисперсия — это математическое ожидание квадрата отклонения от среднего значения.',
            ),
            (
                'Что такое ковариация?',
                'Ковариация показывает, как две случайные величины изменяются совместно.',
            ),
            (
                'Что означает независимость событий?',
                'События независимы, если вероятность их пересечения равна произведению вероятностей.',
            ),
            (
                'Что такое условная вероятность?',
                'Условная вероятность — это вероятность события A при условии, что событие B уже произошло.',
            ),
            (
                'Сформулируйте теорему Байеса.',
                'P(A|B) = P(B|A) * P(A) / P(B), при условии P(B) > 0.',
            ),
            (
                'Что такое функция плотности вероятности?',
                'Функция плотности задает вероятности для непрерывных случайных величин через интегрирование.',
            ),
            (
                'Что такое закон больших чисел?',
                'Выборочные средние стремятся к математическому ожиданию при росте объема выборки.',
            ),
        ]
    else:
        qa = [
            (
                'Что такое задача оптимизации?',
                'Это задача минимизации или максимизации целевой функции при наличии ограничений.',
            ),
            (
                'Что такое выпуклое множество?',
                'Множество называется выпуклым, если отрезок между любыми двумя его точками полностью лежит в этом множестве.',
            ),
            (
                'Что такое выпуклая функция?',
                'Функция является выпуклой, если ее эпиграф образует выпуклое множество.',
            ),
            (
                'Что такое градиент в оптимизации?',
                'Градиент — это вектор частных производных, указывающий направление наискорейшего возрастания.',
            ),
            (
                'Что такое матрица Гессе?',
                'Матрица Гессе — это матрица вторых частных производных функции.',
            ),
            (
                'Что такое условная оптимизация?',
                'Условная оптимизация — это оптимизация при равенствах и/или неравенствах-ограничениях.',
            ),
            (
                'Что такое условия ККТ?',
                'Условия Каруша-Куна-Таккера — это условия первого порядка оптимальности для задач с ограничениями.',
            ),
            (
                'Что такое лагранжиан?',
                'Лагранжиан объединяет целевую функцию и ограничения с помощью множителей.',
            ),
            (
                'Что такое линейный поиск шага?',
                'Линейный поиск выбирает размер шага вдоль направления спуска.',
            ),
            (
                'Что такое стохастический градиентный спуск?',
                'SGD обновляет параметры, используя градиенты, оцененные по мини-батчам.',
            ),
        ]

    lines = [f'# {subject_name}', '']
    for idx, (question, answer) in enumerate(qa, start=1):
        lines.append(f'## Q{idx}: {question}')
        lines.append('A:')
        lines.append(answer)
        lines.append('')
    return '\n'.join(lines).strip() + '\n'


def _subject_file_name(subject_name: str) -> str:
    """Convert subject name to markdown file name."""
    return subject_name.lower().replace(' ', '_') + '.md'


def _normalize_text(value: str) -> str:
    """Normalize text for case-insensitive matching."""
    return ' '.join(value.strip().lower().split())


def _build_enrollments() -> list[tuple[str, str]]:
    """Build deterministic student-subject enrollment list."""
    enrollments: list[tuple[str, str]] = []
    for idx, student in enumerate(STUDENTS):
        if student == 'John':
            enrollments.append((student, 'Machine Learning'))
            continue
        first = SUBJECTS[idx % len(SUBJECTS)]
        second = SUBJECTS[(idx + 1) % len(SUBJECTS)]
        enrollments.append((student, first))
        enrollments.append((student, second))
    return enrollments


def _grade_for(student_idx: int, subject_idx: int, question_id: int) -> int:
    """Create deterministic grade in range 0..100."""
    base = 55 + ((student_idx * 13 + subject_idx * 17 + question_id * 11) % 46)
    if (student_idx + subject_idx + question_id) % 9 == 0:
        base = 35 + ((student_idx + question_id) % 15)
    if subject_idx == 1 and question_id == 4:
        base = 30 + (student_idx % 15)
    return max(0, min(100, base))


def ensure_data_files():
    """Generate data files if they do not exist."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    EXAMS_DIR.mkdir(parents=True, exist_ok=True)

    if not STUDENTS_CSV.exists():
        with STUDENTS_CSV.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['student_name'])
            writer.writeheader()
            for student in STUDENTS:
                writer.writerow({'student_name': student})

    if not ENROLLMENTS_CSV.exists():
        with ENROLLMENTS_CSV.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['student_name', 'subject_name'])
            writer.writeheader()
            for student_name, subject_name in _build_enrollments():
                writer.writerow(
                    {'student_name': student_name, 'subject_name': subject_name}
                )

    if not META_GRADES_CSV.exists():
        with META_GRADES_CSV.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=['student_name', 'subject_name', 'question_id', 'grade'],
            )
            writer.writeheader()
            for student_idx, student in enumerate(STUDENTS):
                student_subjects = subjects_for_student(student)
                for subject in student_subjects:
                    subject_idx = SUBJECTS.index(subject)
                    for question_id in range(1, 11):
                        writer.writerow(
                            {
                                'student_name': student,
                                'subject_name': subject,
                                'question_id': question_id,
                                'grade': _grade_for(
                                    student_idx=student_idx,
                                    subject_idx=subject_idx,
                                    question_id=question_id,
                                ),
                            }
                        )

    for subject in SUBJECTS:
        exam_file = EXAMS_DIR / _subject_file_name(subject)
        exam_file.write_text(_build_exam_markdown(subject), encoding='utf-8')

    logger.debug(f'data_store // data files ready in {DATA_ROOT}')


@lru_cache(maxsize=1)
def load_students() -> list[str]:
    """Load students from generated CSV."""
    ensure_data_files()
    students: list[str] = []
    with STUDENTS_CSV.open('r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            students.append(row['student_name'])
    return students


@lru_cache(maxsize=1)
def load_enrollments() -> dict[str, set[str]]:
    """Load enrollments from generated CSV."""
    ensure_data_files()
    enrollments: dict[str, set[str]] = {}
    with ENROLLMENTS_CSV.open('r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            student_name = row['student_name']
            subject_name = row['subject_name']
            enrollments.setdefault(student_name, set()).add(subject_name)
    return enrollments


@lru_cache(maxsize=1)
def load_meta_grades() -> dict[tuple[str, str], dict[int, int]]:
    """Load question-linked meta-grades for each student and subject."""
    ensure_data_files()
    result: dict[tuple[str, str], dict[int, int]] = {}
    with META_GRADES_CSV.open('r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = (row['student_name'], row['subject_name'])
            result.setdefault(key, {})[int(row['question_id'])] = int(row['grade'])
    return result


def normalize_subject_name(raw_subject: str) -> str | None:
    """Normalize user-provided subject string to canonical subject name."""
    aliases = {
        'machine learning': 'Machine Learning',
        'ml': 'Machine Learning',
        'probability theory': 'Probability Theory',
        'probability': 'Probability Theory',
        'optimization theory': 'Optimization Theory',
        'optimization': 'Optimization Theory',
        'opt theory': 'Optimization Theory',
    }
    normalized = _normalize_text(raw_subject)
    if normalized in aliases:
        return aliases[normalized]
    for subject in SUBJECTS:
        if normalized == _normalize_text(subject):
            return subject
    return None


def normalize_student_name(raw_student: str) -> str | None:
    """Normalize student name to canonical value."""
    target = _normalize_text(raw_student)
    for student in load_students():
        if _normalize_text(student) == target:
            return student
    return None


def subjects_for_student(student_name: str) -> list[str]:
    """Get enrolled subjects for student name."""
    if student_name == 'John':
        return ['Machine Learning']
    student_idx = STUDENTS.index(student_name)
    first = SUBJECTS[student_idx % len(SUBJECTS)]
    second = SUBJECTS[(student_idx + 1) % len(SUBJECTS)]
    return [first, second]
