"""Tests for FAISS index building."""

from __future__ import annotations

import json

import faiss
import pytest

import src.prepare_data as prepare_data

pytestmark = [pytest.mark.skip]


def test_build_faiss_index_runs_successfully():
    """build_faiss_index uses real markdown and writes index artifacts."""
    prepare_data.FAISS_INDEX_PATH.unlink(missing_ok=True)
    prepare_data.RAG_CHUNKS_PATH.unlink(missing_ok=True)

    result_path = prepare_data.build_faiss_index()
    expected_chunks_count = len(prepare_data._load_markdown_chunks())

    assert result_path == prepare_data.FAISS_INDEX_PATH
    assert prepare_data.FAISS_INDEX_PATH.exists()
    assert prepare_data.RAG_CHUNKS_PATH.exists()

    index = faiss.read_index(str(prepare_data.FAISS_INDEX_PATH))
    rag_chunks_payload = json.loads(
        prepare_data.RAG_CHUNKS_PATH.read_text(encoding='utf-8')
    )

    assert index.ntotal == expected_chunks_count
    assert len(rag_chunks_payload['chunks']) == expected_chunks_count
