"""Smoke tests for build_index helpers."""

from pathlib import Path

import pytest

from src.s02_simple_haiku.rag import build_index as bi

pytestmark = pytest.mark.unit


def fake_post_embeddings(payload: dict) -> dict:
    """Return deterministic embeddings for tests."""
    inputs = payload.get('input', [])
    data = [
        {'index': idx, 'embedding': [0.1, 0.2, 0.3]} for idx, _ in enumerate(inputs)
    ]
    return {'data': data}


def test_split_sentences_basic():
    """Split text into sentences."""
    text = 'Первое предложение. Второе? Третье!'
    result = bi.split_sentences(text)
    assert result == ['Первое предложение.', 'Второе?', 'Третье!']


def test_chunk_text_respects_limit():
    """Chunking should respect max size."""
    text = 'Раз. Два. Три.'
    chunks = bi.chunk_text(text, max_chunk_chars=6)
    assert chunks
    assert all(len(chunk) <= 6 for chunk in chunks)


def test_extract_title_and_fallback():
    """Extract first non-empty line or fallback."""
    text = '\n\nЗаголовок\nДальше текст'
    assert bi.extract_title(text) == 'Заголовок'
    assert bi.extract_title('\n\n') == 'Без заголовка'


def test_load_markdown_chunks(tmp_path: Path):
    """Load markdown files into RagChunk list."""
    file_path = tmp_path / 'doc.md'
    file_path.write_text('Тестовый заголовок\nТекст. Еще текст.', encoding='utf-8')
    chunks = bi.load_markdown_chunks(tmp_path, max_chunk_chars=20)
    assert chunks
    assert chunks[0].source == 'doc.md'
    assert chunks[0].title.startswith('Тестовый заголовок')


def test_load_markdown_chunks_multiple_docs(tmp_path: Path):
    """Load multiple markdown files into RagChunk list."""
    (tmp_path / 'a.md').write_text('A title\nOne. Two.', encoding='utf-8')
    (tmp_path / 'b.md').write_text('B title\nAlpha. Beta.', encoding='utf-8')
    (tmp_path / 'c.md').write_text('C title\nFirst. Second.', encoding='utf-8')

    chunks = bi.load_markdown_chunks(tmp_path, max_chunk_chars=30)
    assert chunks
    sources = {chunk.source for chunk in chunks}
    assert sources == {'a.md', 'b.md', 'c.md'}


def test_build_and_init_faiss_index(tmp_path: Path, monkeypatch):
    """Build and init FAISS index with mocked embeddings."""
    monkeypatch.setattr(bi, 'post_embeddings', fake_post_embeddings)
    monkeypatch.setattr(bi, 'INDEX_PATH', tmp_path / 'faiss.index')

    index, dim = bi.build_faiss_index(['a', 'b'])
    assert index is not None
    assert dim == 3
    assert index.ntotal == 2

    index, dim = bi.init_faiss_index(['a', 'b'])
    assert index is not None
    assert dim == 3
    assert bi.INDEX_PATH.exists()


def test_init_faiss_index_rebuilds_when_chunk_count_mismatch(
    tmp_path: Path, monkeypatch
):
    """Index is rebuilt when index.ntotal != len(texts)."""
    monkeypatch.setattr(bi, 'post_embeddings', fake_post_embeddings)
    monkeypatch.setattr(bi, 'INDEX_PATH', tmp_path / 'faiss.index')

    index_old, _ = bi.build_faiss_index(['a', 'b'])
    bi.save_faiss_index(index_old)
    bi.save_index_hash(bi.compute_texts_hash(['a', 'b']))
    assert bi.INDEX_PATH.exists()

    index, dim = bi.init_faiss_index(['a', 'b', 'c', 'd'])
    assert index is not None
    assert index.ntotal == 4
    assert dim == 3


def test_init_faiss_index_rebuilds_when_texts_content_changes(
    tmp_path: Path, monkeypatch
):
    """Index is rebuilt when texts content changes but count stays the same."""
    monkeypatch.setattr(bi, 'post_embeddings', fake_post_embeddings)
    monkeypatch.setattr(bi, 'INDEX_PATH', tmp_path / 'faiss.index')

    index_old, _ = bi.build_faiss_index(['a', 'b'])
    bi.save_faiss_index(index_old)
    bi.save_index_hash(bi.compute_texts_hash(['a', 'b']))

    index, dim = bi.init_faiss_index(['a', 'x'])  # same count, different content
    assert index is not None
    assert index.ntotal == 2
    assert dim == 3
    assert bi.load_index_hash() == bi.compute_texts_hash(['a', 'x'])
