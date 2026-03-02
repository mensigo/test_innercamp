"""Public API functions for student data access."""

from .get_avg_overall_score import get_avg_overall_score
from .get_avg_score import get_avg_score
from .get_top_students import get_top_students
from .search_rag import search_rag

__all__ = [
    'get_top_students',
    'get_avg_score',
    'get_avg_overall_score',
    'search_rag',
]
