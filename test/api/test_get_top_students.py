from __future__ import annotations

import pytest

from src.api.get_top_students import get_top_students
from src.prepare_data import ensure_students_csv

pytestmark = [pytest.mark.api, pytest.mark.unit]


@pytest.mark.parametrize(
    ('subject_name', 'k'),
    [
        ('Machine Learning', 3),
        ('Probability Theory', 5),
        ('Optimization Theory', 7),
    ],
)
def test_get_top_students_returns_sorted_top_k(subject_name: str, k: int):
    ensure_students_csv()

    result = get_top_students(subject_name, k=k)

    assert len(result) == k
    scores = [float(item['score']) for item in result]
    assert scores == sorted(scores, reverse=True)
    assert all(item['name'].strip() for item in result)


def test_get_top_students_returns_empty_for_invalid_subject():
    ensure_students_csv()

    assert get_top_students('', k=3) == []
    assert get_top_students('  ', k=4) == []
    assert get_top_students('Unknown Subject', k=5) == []
    assert get_top_students('optimization theory', k=6) == []


def test_get_top_students_returns_empty_for_k_less_than_one():
    ensure_students_csv()

    assert get_top_students('Optimization Theory', k=0) == []
    assert get_top_students('Optimization Theory', k=-5) == []


def test_get_top_students_raises_for_k_greater_than_ten():
    ensure_students_csv()

    with pytest.raises(ValueError):
        get_top_students('Optimization Theory', k=11)
