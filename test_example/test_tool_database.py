"""Focused tests for database_tool top_students operation."""

from __future__ import annotations

import csv

import pytest

from src_example.data_store import ENROLLMENTS_CSV, META_GRADES_CSV, SUBJECTS
from src_example.tools.database_tool import database_tool


def _expected_top_students(subject_name: str, top_k: int) -> list[dict]:
    """Compute top students for a subject directly from generated CSV files."""
    enrolled_students: set[str] = set()
    with ENROLLMENTS_CSV.open('r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['subject_name'] == subject_name:
                enrolled_students.add(row['student_name'])

    grades_by_student: dict[str, list[int]] = {}
    with META_GRADES_CSV.open('r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            student_name = row['student_name']
            if student_name not in enrolled_students:
                continue
            if row['subject_name'] != subject_name:
                continue
            grades_by_student.setdefault(student_name, []).append(int(row['grade']))

    rows: list[dict] = []
    for student_name, grades in grades_by_student.items():
        if not grades:
            continue
        avg_grade = round(sum(grades) / len(grades), 1)
        rows.append({'name': student_name, 'grade': avg_grade})

    rows.sort(key=lambda row: (-row['grade'], row['name']))
    return rows[:top_k]


@pytest.mark.parametrize(
    ('subject_name', 'top_k'),
    [
        (SUBJECTS[0], 3),
        (SUBJECTS[1], 5),
        (SUBJECTS[2], 7),
    ],
)
def test_database_tool_top_students_matches_generated_data(
    subject_name: str, top_k: int
):
    result = database_tool(
        operation='top_students',
        subject_name=subject_name,
        top_k=top_k,
    )

    assert 'error' not in result
    assert result['operation'] == 'top_students'
    assert result['subject_name'] == subject_name

    expected_rows = _expected_top_students(subject_name=subject_name, top_k=top_k)
    assert result['data'] == expected_rows
