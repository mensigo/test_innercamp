"""Prepare deterministic students CSV for API and future RAG steps."""

from __future__ import annotations

import csv
import random
from pathlib import Path

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

DATA_DIR = Path(__file__).resolve().parent / 'data'
STUDENTS_CSV = DATA_DIR / 'students.csv'
DEFAULT_SEED = 42


def _subjects_for_student(student_idx: int, student_name: str) -> list[str]:
    """Return 1-2 subjects assigned to student."""
    if student_name == 'John':
        return ['Machine Learning']
    first = SUBJECTS[student_idx % len(SUBJECTS)]
    second = SUBJECTS[(student_idx + 1) % len(SUBJECTS)]
    return [first, second]


def _score_for(student_idx: int, subject_idx: int, seed: int) -> float:
    """Build deterministic seed-based score in range 3.0..5.0."""
    rng = random.Random(seed + student_idx * 101 + subject_idx * 37)
    score = rng.uniform(3.0, 5.0)
    return round(score, 1)


def build_students_rows(seed: int = DEFAULT_SEED) -> list[dict[str, str]]:
    """Build rows for students.csv with expected schema."""
    rows: list[dict[str, str]] = []
    for student_idx, student_name in enumerate(STUDENTS):
        for subject_name in _subjects_for_student(student_idx, student_name):
            subject_idx = SUBJECTS.index(subject_name)
            rows.append(
                {
                    'student_name': student_name,
                    'subject_name': subject_name,
                    'score': f'{_score_for(student_idx, subject_idx, seed):.1f}',
                }
            )
    return rows


def ensure_students_csv(force: bool = False, seed: int = DEFAULT_SEED) -> Path:
    """Create students.csv when missing or when force=True."""
    if STUDENTS_CSV.exists() and not force:
        return STUDENTS_CSV

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with STUDENTS_CSV.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=['student_name', 'subject_name', 'score'],
        )
        writer.writeheader()
        for row in build_students_rows(seed=seed):
            writer.writerow(row)
    return STUDENTS_CSV


if __name__ == '__main__':
    ensure_students_csv(force=True, seed=DEFAULT_SEED)
