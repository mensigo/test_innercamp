"""FAISS index build and persistence helpers."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from src import post_embeddings

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
INDEX_PATH = BASE_DIR / 'faiss.index'
DEFAULT_MAX_CHUNK_CHARS = 800


def _hash_path() -> Path:
    """Path to stored texts hash (derived from INDEX_PATH for testability)."""
    return INDEX_PATH.parent / (INDEX_PATH.stem + '.hash')


def compute_texts_hash(texts: list[str]) -> str:
    """
    Compute SHA256 hash of texts for index consistency validation.
    """
    content = '\n'.join(texts)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


@dataclass
class RagChunk:
    text: str
    source: str
    title: str
    ordinal: int


def split_sentences(text: str) -> list[str]:
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
    Split markdown text into smaller chunks by sentences.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ''

    for sentence in sentences:
        if not sentence:
            continue
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
    Extract the first non-empty line from markdown text.
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


def load_markdown_chunks(data_dir: Path, max_chunk_chars: int) -> list[RagChunk]:
    """
    Load markdown files from data dir and split into chunks.
    """
    if not data_dir.exists():
        return []

    chunks: list[RagChunk] = []
    for path in sorted(data_dir.glob('*.md')):
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

    return chunks


def build_rag_chunks(
    data_dir: Path | None = None,
    max_chunk_chars: int | None = None,
) -> list[RagChunk]:
    """
    Build rag chunks from markdown data directory.
    """
    resolved_dir = data_dir or DATA_DIR
    resolved_max = max_chunk_chars or DEFAULT_MAX_CHUNK_CHARS
    return load_markdown_chunks(resolved_dir, resolved_max)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using /embeddings.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {'input': batch}
        response = post_embeddings(payload)

        if 'error' in response:
            print(f'RAG embeddings error: {response["error"]}')
            return []

        data = response.get('data', [])
        data_sorted = sorted(data, key=lambda item: item.get('index', 0))
        embeddings = [item.get('embedding', []) for item in data_sorted]
        all_embeddings.extend(embeddings)

    return all_embeddings


def build_faiss_index(texts: list[str]) -> tuple[faiss.Index | None, int]:
    """
    Build FAISS index for provided texts.
    """
    if not texts:
        return None, 0

    embeddings = embed_texts(texts)
    if not embeddings:
        return None, 0

    vectors = np.array(embeddings, dtype='float32')
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index, dimension


def load_faiss_index() -> tuple[faiss.Index | None, int]:
    """
    Load FAISS index from disk if available.
    """
    if not INDEX_PATH.exists():
        return None, 0
    try:
        index = faiss.read_index(str(INDEX_PATH))
        return index, index.d
    except Exception as ex:
        print(f'RAG: не удалось загрузить индекс: {ex}')
        return None, 0


def save_faiss_index(index: faiss.Index):
    """
    Save FAISS index to disk.
    """
    try:
        faiss.write_index(index, str(INDEX_PATH))
    except Exception as ex:
        print(f'RAG: не удалось сохранить индекс: {ex}')


def load_index_hash() -> str | None:
    """
    Load stored texts hash if available.
    """
    path = _hash_path()
    if not path.exists():
        return None
    try:
        return path.read_text(encoding='utf-8').strip()
    except Exception:
        return None


def save_index_hash(hash_str: str):
    """
    Save texts hash for index consistency validation.
    """
    try:
        _hash_path().write_text(hash_str, encoding='utf-8')
    except Exception as ex:
        print(f'RAG: не удалось сохранить хеш индекса: {ex}')


def init_faiss_index(texts: list[str]) -> tuple[faiss.Index | None, int]:
    """
    Initialize FAISS index with load or rebuild.
    Rebuilds when count or content hash of texts differs from saved index.
    """
    if not texts:
        return None, 0

    texts_hash = compute_texts_hash(texts)
    index, dimension = load_faiss_index()

    if index is not None:
        stored_hash = load_index_hash()
        count_ok = index.ntotal == len(texts)
        hash_ok = stored_hash == texts_hash
        if not count_ok or not hash_ok:
            reason = 'количество чанков' if not count_ok else 'содержимое чанков'
            print(f'RAG: индекс не соответствует ({reason}), пересоздание')
            index, dimension = build_faiss_index(texts)
            if index is not None:
                save_faiss_index(index)
                save_index_hash(texts_hash)
    else:
        index, dimension = build_faiss_index(texts)
        if index is not None:
            save_faiss_index(index)
            save_index_hash(texts_hash)

    return index, dimension
