"""FAISS index build and persistence helpers."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from src import post_embeddings

from .logger import logger

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
INDEX_PATH = BASE_DIR / 'faiss.index'

DEFAULT_MAX_CHUNK_CHARS = 800
EMBED_BATCH_SIZE = 8


@dataclass
class RagChunk:
    text: str
    source: str
    title: str
    ordinal: int


def _hash_path() -> Path:
    """Path to stored texts hash (derived from INDEX_PATH for testability)."""
    return INDEX_PATH.parent / (INDEX_PATH.stem + '.hash')


def _compute_texts_hash(texts: list[str]) -> str:
    """
    Compute SHA256 hash of texts for index consistency validation.
    """
    if not texts:
        raise RuntimeError('build_index // No texts to compute hash')

    content = '\n'.join(texts)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation boundaries.
    """
    cleaned = ' '.join(text.split())
    if not cleaned:
        return []
    parts = re.split(r'(?<=[.!?])\s+', cleaned)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_chunk_chars: int) -> list[str]:
    """
    Split Markdown text into smaller chunks by sentences.
    """
    sentences = _split_sentences(text)
    if not sentences:
        raise RuntimeError('build_index // No sentences to chunk')

    chunks: list[str] = []
    current = ''

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chunk_chars:
            if current:
                chunks.append(current.strip())
                current = ''

            words = sentence.split()
            temp = ''

            for word in words:
                if len(temp) + len(word) + 1 > max_chunk_chars:
                    if temp:
                        chunks.append(temp.strip())
                    temp = word
                else:
                    temp = f'{temp} {word}' if temp else word

            if temp:
                chunks.append(temp.strip())

            continue

        if not current:
            current = sentence
            continue

        candidate = f'{current} {sentence}'
        if len(candidate) > max_chunk_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current.strip())

    return chunks


def extract_title(text: str) -> str:
    """
    Extract the first non-empty line from Markdown text.
    """
    return next(
        (line.strip() for line in text.splitlines() if line.strip()),
        'Без заголовка',
    )


def build_chunk_title(base_title: str, ordinal: int, total: int) -> str:
    """
    Build chunk title with ordinal if needed.
    """
    if total <= 1:
        return base_title
    return f'{base_title} ({ordinal})'


def build_rag_chunks(
    data_dir: Path | None = None,
    max_chunk_chars: int | None = None,
) -> list[RagChunk]:
    """
    Load Markdown files from data dir and split into chunks.
    """
    data_dir = data_dir or DATA_DIR
    max_chunk_chars = max_chunk_chars or DEFAULT_MAX_CHUNK_CHARS

    if not data_dir.exists():
        raise RuntimeError(
            f'build_index // No files to load (path does not exist): {data_dir}'
        )

    path_lst = sorted(data_dir.glob('*.md'))
    if not path_lst:
        raise RuntimeError(f'build_index // No files to load (empty dir): {data_dir}')

    chunks: list[RagChunk] = []
    for path in tqdm(path_lst):
        text = path.read_text(encoding='utf-8')
        base_title = extract_title(text)
        raw_chunks = chunk_text(text, max_chunk_chars)
        total = len(raw_chunks)
        for idx, chunk in enumerate(raw_chunks, start=1):
            title = build_chunk_title(base_title, idx, total)
            chunks.append(
                RagChunk(
                    text=chunk,
                    source=path.name,
                    title=title,
                    ordinal=idx,
                )
            )
    if not chunks:
        raise RuntimeError('build_index // No chunks built')

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using /embeddings.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        payload = {'input': batch}
        response = post_embeddings(payload)

        if 'error' in response:
            raise RuntimeError(
                'build_index // embeddings error: {}'.format(response['error'])
            )

        data = response.get('data', [])
        data_sorted = sorted(data, key=lambda item: item.get('index', 0))
        embeddings = [item.get('embedding', []) for item in data_sorted]
        all_embeddings.extend(embeddings)

    return all_embeddings


def _build_faiss_index(texts: list[str]) -> faiss.Index | None:
    """
    Build FAISS index for provided texts.
    """
    if not texts:
        raise RuntimeError('build_index // No texts to build index')

    embeddings = embed_texts(texts)
    if not embeddings:
        raise RuntimeError('build_index // No embeddings to build index')

    vectors = np.array(embeddings, dtype='float64')
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def load_faiss_index() -> faiss.Index | None:
    """
    Load FAISS index from disk if available.
    """
    if not INDEX_PATH.exists():
        logger.warning(
            f'build_index // No index to load (path does not exist): {INDEX_PATH}'
        )
        return None

    try:
        index = faiss.read_index(str(INDEX_PATH))
        return index

    except Exception as ex:
        raise RuntimeError(f'build_index // Failed to load index: {ex}') from ex


def save_faiss_index(index: faiss.Index):
    """
    Save FAISS index to disk.
    """
    try:
        faiss.write_index(index, str(INDEX_PATH))
    except Exception as ex:
        raise RuntimeError(f'build_index // Failed to save index: {ex}') from ex


def load_index_hash() -> str | None:
    """
    Load stored texts hash if available.
    """
    path = _hash_path()
    if not path.exists():
        logger.warning(f'build_index // No hash to load (path does not exist): {path}')
        return None

    try:
        return path.read_text(encoding='utf-8').strip()

    except Exception as ex:
        raise RuntimeError(f'build_index // Failed to load hash: {ex}') from ex


def save_index_hash(hash_str: str):
    """
    Save texts hash for index consistency validation.
    """
    try:
        _hash_path().write_text(hash_str, encoding='utf-8')
    except Exception as ex:
        raise RuntimeError(f'build_index // Failed to save hash: {ex}') from ex


def init_faiss_index(texts: list[str]) -> faiss.Index | None:
    """
    Initialize FAISS index with load or rebuild.
    Rebuilds when count or content hash of texts differs from saved index.
    """
    if not texts:
        raise RuntimeError('build_index // No texts to init index')

    texts_hash = _compute_texts_hash(texts)
    index = load_faiss_index()

    if index is not None:
        stored_hash = load_index_hash()
        count_ok = index.ntotal == len(texts)
        hash_ok = stored_hash == texts_hash

        if count_ok and hash_ok:
            return index

        reason = 'num of chunks' if not count_ok else "chunks' content"
        logger.info(f'build_index // Rebuilding index (bcs {reason} changed)')

    index = _build_faiss_index(texts)
    save_faiss_index(index)
    save_index_hash(texts_hash)

    return index
