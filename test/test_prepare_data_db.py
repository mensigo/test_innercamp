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
