from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

from src.api.get_avg_overall_score import get_avg_overall_score
from src.prepare_data import ensure_students_csv

pytestmark = [pytest.mark.api, pytest.mark.unit]

get_avg_overall_score_module = importlib.import_module('src.api.get_avg_overall_score')
STUDENTS_CSV = Path(__file__).resolve().parents[2] / 'src' / 'data' / 'students.csv'


def _compute_overall_average() -> float:
    scores: list[float] = []
    with STUDENTS_CSV.open('r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            scores.append(float(row['score']))
    return round(sum(scores) / len(scores), 1)


def test_get_avg_overall_score_returns_average_with_one_decimal():
    ensure_students_csv(force=True, seed=42)

    result = get_avg_overall_score()
    expected_avg_score = _compute_overall_average()

    assert result == {'avg_score': expected_avg_score}


def test_get_avg_overall_score_returns_empty_when_no_data(monkeypatch):
    monkeypatch.setattr(
        get_avg_overall_score_module,
        'load_students_rows',
        lambda: [],
    )

    assert get_avg_overall_score() == {}
