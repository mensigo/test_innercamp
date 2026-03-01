"""API function placeholder for RAG search."""

from __future__ import annotations


def search_rag(query: str, k: int = 2) -> dict[str, list[str]]:
    """Return top-k chunks for a query from RAG index."""
    if not query:
        return {'chunks': []}
    if k < 1:
        return {'chunks': []}
    return {'chunks': []}
