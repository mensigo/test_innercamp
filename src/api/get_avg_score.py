"""API function for average score by subject."""

from __future__ import annotations

from ._common import load_students_rows


def get_avg_score(subject_name: str) -> dict[str, float]:
    """Return average score for selected subject."""
    if not subject_name:
        return {}

    scores: list[float] = []
    for row in load_students_rows():
        if row.get('subject_name') == subject_name:
            scores.append(float(row['score']))

    if not scores:
        return {}
    return {'avg_score': round(sum(scores) / len(scores), 1)}
