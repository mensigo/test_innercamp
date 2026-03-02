from __future__ import annotations

import pytest

from src.api.get_top_students import get_top_students
from src.prepare_data import ensure_students_csv

pytestmark = [pytest.mark.api, pytest.mark.unit]


def test_get_top_students_returns_sorted_top_k():
    ensure_students_csv(force=True, seed=42)

    result = get_top_students('Optimization Theory', k=5)

    assert 0 < len(result) <= 5
    scores = [float(item['score']) for item in result]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(item['name'], str) for item in result)


def test_get_top_students_returns_empty_for_invalid_subject():
    ensure_students_csv(force=True, seed=42)

    assert get_top_students('', k=3) == []
    assert get_top_students('Unknown Subject', k=3) == []
    assert get_top_students('optimization theory', k=3) == []


def test_get_top_students_returns_empty_for_k_less_than_one():
    ensure_students_csv(force=True, seed=42)

    assert get_top_students('Optimization Theory', k=0) == []


def test_get_top_students_raises_for_k_greater_than_ten():
    ensure_students_csv(force=True, seed=42)

    with pytest.raises(ValueError):
        get_top_students('Optimization Theory', k=11)
