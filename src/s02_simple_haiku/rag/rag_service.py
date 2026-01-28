"""Flask RAG service with FAISS index."""

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from flask import Flask, jsonify, request

from src import config
from src.utils_openai import post_embeddings

DATA_DIR = Path(__file__).resolve().parent / 'data'
DEFAULT_TOP_K = 3
MAX_CHUNK_CHARS = 800

app = Flask(__name__)


@dataclass
class RagChunk:
    text: str
    source: str


RAG_CHUNKS: list[RagChunk] = []
RAG_INDEX = None
RAG_DIM = 0


def chunk_text(text: str, source: str) -> list[RagChunk]:
    """
    Split markdown text into smaller chunks by paragraphs.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks: list[RagChunk] = []

    for paragraph in paragraphs:
        if len(paragraph) <= MAX_CHUNK_CHARS:
            chunks.append(RagChunk(text=paragraph, source=source))
            continue

        start = 0
        while start < len(paragraph):
            end = start + MAX_CHUNK_CHARS
            chunk = paragraph[start:end].strip()
            if chunk:
                chunks.append(RagChunk(text=chunk, source=source))
            start = end

    return chunks


def load_markdown_chunks(data_dir: Path) -> list[RagChunk]:
    """
    Load markdown files from data dir and split into chunks.
    """
    if not data_dir.exists():
        return []

    chunks: list[RagChunk] = []
    for path in sorted(data_dir.glob('*.md')):
        text = path.read_text(encoding='utf-8')
        chunks.extend(chunk_text(text, source=path.name))

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using OpenRouter embeddings API.
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


def build_faiss_index(chunks: list[RagChunk]):
    """
    Build FAISS index for provided chunks.
    """
    if not chunks:
        return None, 0

    embeddings = embed_texts([chunk.text for chunk in chunks])
    if not embeddings:
        return None, 0

    vectors = np.array(embeddings, dtype='float32')
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index, dimension


def init_rag_index():
    """
    Initialize RAG chunks and FAISS index on service start.
    """
    global RAG_CHUNKS, RAG_INDEX, RAG_DIM
    RAG_CHUNKS = load_markdown_chunks(DATA_DIR)
    RAG_INDEX, RAG_DIM = build_faiss_index(RAG_CHUNKS)

    if not RAG_CHUNKS:
        print('RAG: нет markdown данных для индексации')
    elif RAG_INDEX is None:
        print('RAG: индекс не создан, проверьте доступ к embeddings')
    else:
        print(f'RAG: индекс создан, чанков={len(RAG_CHUNKS)}')


def search_chunks(question: str, top_k: int) -> list[dict]:
    """
    Search similar chunks for the given question.
    """
    if not question or RAG_INDEX is None:
        return []

    embeddings = embed_texts([question])
    if not embeddings:
        return []

    query_vector = np.array(embeddings, dtype='float32')
    _, indices = RAG_INDEX.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(RAG_CHUNKS):
            continue
        chunk = RAG_CHUNKS[idx]
        results.append({'text': chunk.text, 'source': chunk.source})

    return results


@app.route('/search', methods=['POST'])
def search():
    """
    Search RAG index by question and return top chunks.
    """
    payload = request.get_json()
    if not payload or 'question' not in payload:
        return jsonify({'error': 'Missing question'}), 400

    question = str(payload.get('question', '')).strip()
    top_k = int(payload.get('top_k', DEFAULT_TOP_K))
    top_k = max(1, min(top_k, 5))

    results = search_chunks(question, top_k)
    return jsonify({'results': results, 'question': question, 'top_k': top_k})


init_rag_index()


if __name__ == '__main__':
    app.run(
        host='localhost',
        port=config.tool_rag_port,
        debug=config.flask_debug,
    )
