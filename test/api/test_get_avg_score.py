from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

from src.api.get_avg_score import get_avg_score
from src.prepare_data import ensure_students_csv

pytestmark = [pytest.mark.api, pytest.mark.unit]

get_avg_score_module = importlib.import_module('src.api.get_avg_score')
STUDENTS_CSV = Path(__file__).resolve().parents[2] / 'src' / 'data' / 'students.csv'


def _compute_subject_average(subject_name: str) -> float:
    scores: list[float] = []
    with STUDENTS_CSV.open('r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['subject_name'] == subject_name:
                scores.append(float(row['score']))
    return round(sum(scores) / len(scores), 1)


def test_get_avg_score_returns_average_with_one_decimal():
    ensure_students_csv(force=True, seed=42)

    result = get_avg_score('Machine Learning')
    expected_avg_score = _compute_subject_average('Machine Learning')

    assert result == {'avg_score': expected_avg_score}


def test_get_avg_score_returns_empty_for_invalid_subject():
    ensure_students_csv(force=True, seed=42)

    assert get_avg_score('') == {}
    assert get_avg_score('Unknown Subject') == {}
    assert get_avg_score('machine learning') == {}


def test_get_avg_score_returns_empty_when_no_data(monkeypatch):
    monkeypatch.setattr(get_avg_score_module, 'load_students_rows', lambda: [])

    assert get_avg_score('Machine Learning') == {}
