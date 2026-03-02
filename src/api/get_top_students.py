"""API function for top students by subject."""

from __future__ import annotations

from ._common import load_students_rows


def get_top_students(subject_name: str, k: int = 3) -> list[dict[str, float | str]]:
    """Return top-k students for selected subject."""
    if not subject_name:
        return []
    if k < 1:
        return []
    if k > 10:
        raise ValueError('k must be <= 10')

    matches: list[dict[str, float | str]] = []
    for row in load_students_rows():
        if row.get('subject_name') != subject_name:
            continue
        score = float(row['score'])
        matches.append({'name': row['student_name'], 'score': score})

    matches.sort(key=lambda item: (-float(item['score']), item['name']))
    return matches[:k]
