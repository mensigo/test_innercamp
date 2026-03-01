"""Public API functions for student data access."""

from .get_avg_overall_score import get_avg_overall_score
from .get_avg_score import get_avg_score
from .get_students import get_students
from .search_rag import search_rag

__all__ = [
    'get_students',
    'get_avg_score',
    'get_avg_overall_score',
    'search_rag',
]
