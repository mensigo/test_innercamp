"""API function for overall average score."""

from __future__ import annotations

from ._common import load_students_rows


def get_avg_overall_score() -> dict[str, float]:
    """Return average score across all subjects."""
    scores = [float(row['score']) for row in load_students_rows()]
    if not scores:
        return {}
    return {'avg_score': round(sum(scores) / len(scores), 1)}
