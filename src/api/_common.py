"""Shared helpers for working with students CSV."""

from __future__ import annotations

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
STUDENTS_CSV = DATA_DIR / 'students.csv'


def load_students_rows() -> list[dict[str, str]]:
    """Load all rows from students.csv."""
    if not STUDENTS_CSV.exists():
        return []
    with STUDENTS_CSV.open('r', encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))
