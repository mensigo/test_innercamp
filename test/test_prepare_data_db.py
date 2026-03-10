"""Tests for deterministic students data preparation."""

from __future__ import annotations

import pytest

from src.prepare_data import build_students_rows

pytestmark = [pytest.mark.unit]


def test_build_students_rows_is_seed_deterministic():
    """Same seed should generate identical rows."""
    rows_a = build_students_rows(seed=42)
    rows_b = build_students_rows(seed=42)
    rows_c = build_students_rows(seed=777)

    assert rows_a == rows_b
    assert rows_a != rows_c


def test_students_data_has_expected_shape_and_score_format():
    """Students data has 30 unique students and valid score format."""
    rows = build_students_rows()

    subjects_by_student: dict[str, set[str]] = {}
    for row in rows:
        student_name = row['student_name']
        subject_name = row['subject_name']
        score = row['score']

        subjects_by_student.setdefault(student_name, set()).add(subject_name)

        score_value = float(score)
        assert 3.0 <= score_value <= 5.0
        integer_part, fractional_part = score.split('.')
        assert integer_part in {'3', '4', '5'}
        assert len(fractional_part) == 1 and fractional_part.isdigit()

    assert len(subjects_by_student) == 30
    assert all(len(subjects) == 2 for subjects in subjects_by_student.values())


def test_top10_optimization_has_student_above_overall_average():
    """Top-10 Optimization contains at least one score above overall average."""
    rows = build_students_rows()

    all_scores = [float(row['score']) for row in rows]
    overall_avg_score = sum(all_scores) / len(all_scores)

    optimization_rows = [
        row for row in rows if row['subject_name'] == 'Optimization Theory'
    ]
    optimization_rows.sort(key=lambda row: float(row['score']), reverse=True)
    top10_optimization = optimization_rows[:10]

    assert any(float(row['score']) > overall_avg_score for row in top10_optimization)
