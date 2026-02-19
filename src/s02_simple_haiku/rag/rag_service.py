"""Flask RAG service with FAISS index."""

import numpy as np
from flask import Flask, jsonify, request

from src import config, post_chat_completions

from .build_index import (
    RagChunk,
    build_rag_chunks,
    embed_texts,
    init_faiss_index,
)
from .logger import logger

DEFAULT_TOP_K = 2

app = Flask(__name__)


RAG_CHUNKS: list[RagChunk] = []
RAG_INDEX = None
RAG_DIM = 0
_RAG_INITIALIZED = False


def init_rag_index():
    """
    Initialize RAG chunks and FAISS index on service start.
    """
    global RAG_CHUNKS, RAG_INDEX, RAG_DIM
    RAG_CHUNKS = build_rag_chunks()

    if not RAG_CHUNKS:
        logger.critical('rag_service // нет markdown данных для индексации')
        return

    texts = [chunk.text for chunk in RAG_CHUNKS]
    RAG_INDEX, RAG_DIM = init_faiss_index(texts)

    if RAG_INDEX is None:
        logger.critical(
            'rag_service // индекс не создан, проверьте доступ к embeddings'
        )
    else:
        logger.info(f'rag_service // индекс создан, чанков={len(RAG_CHUNKS)}')


def _ensure_rag_index():
    """Lazy init: build index on first request."""
    global _RAG_INITIALIZED
    if _RAG_INITIALIZED or RAG_INDEX is not None:
        return
    _RAG_INITIALIZED = True
    init_rag_index()


@app.before_request
def _before_request():
    """Ensure RAG index is built before handling requests."""
    try:
        _ensure_rag_index()
    except Exception as ex:
        logger.error(f'rag_service // init error: {ex}')
        return jsonify({'error': 'Initialization failed'}), 500


def search_chunks(question: str, top_k: int) -> list[RagChunk]:
    """
    Search similar chunks for the given question.
    """
    if not question or RAG_INDEX is None:
        return []

    embeddings = embed_texts([question])
    if not embeddings:
        return []

    query_vector = np.array(embeddings, dtype='float32')
    max_k = min(top_k, len(RAG_CHUNKS))
    if max_k <= 0:
        return []

    _, indices = RAG_INDEX.search(query_vector, max_k)

    results: list[RagChunk] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(RAG_CHUNKS):
            continue
        results.append(RAG_CHUNKS[idx])

    return results


def format_context(chunks: list[RagChunk]) -> str:
    """
    Build context block from chunks.
    """
    context_lines = []
    for chunk in chunks:
        context_lines.append(f'Источник: {chunk.title}\n{chunk.text}')
    return '\n\n'.join(context_lines)


def answer_with_context(question: str, chunks: list[RagChunk]) -> str:
    """
    Ask LLM using retrieved chunks.
    """
    if not chunks:
        return 'Не удалось получить контекст из базы знаний.'

    system_prompt = """Ты помощник по японской поэзии. Отвечай строго по контексту.
Если контекста недостаточно, скажи "данных нет"."""
    user_prompt = (
        f'Вопрос: {question}\n\nКонтекст:\n{format_context(chunks)}\n\n'
        'Ответь кратко и по делу.'
    )

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': config.freezing,
    }

    response = post_chat_completions(payload)
    if 'error' in response:
        logger.error(f'rag_service // LLM Error: {response["error"]}')
        return 'Ошибка при обращении к LLM.'

    try:
        return response['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError) as ex:
        logger.error(f'rag_service // LLM Response Error: {ex}')
        raise Exception('Ошибка при обработке ответа LLM.') from ex


@app.route('/search', methods=['POST'])
def search():
    """
    Search RAG index by question and return top chunks.
    """
    payload = request.get_json()

    if not payload or 'question' not in payload:
        logger.error('rag_service // search error: missing question')
        return jsonify({'error': 'Missing question'}), 400

    try:
        question = str(payload.get('question', '')).strip()
        top_k = int(payload.get('top_k', DEFAULT_TOP_K))
        top_k = max(1, min(top_k, DEFAULT_TOP_K))

        chunks = search_chunks(question, top_k)
        answer = answer_with_context(question, chunks)
        chunk_titles = [chunk.title for chunk in chunks]
        chunk_texts = [chunk.text for chunk in chunks]

        logger.info(
            f'rag_service // search ok, question_len={len(question)}, top_k={top_k}, chunks={len(chunks)}'
        )
        return jsonify(
            {
                'answer': answer,
                'chunk_title_list': chunk_titles,
                'chunk_texts': chunk_texts,
                'question': question,
                'top_k': top_k,
            }
        )
    except RuntimeError as ex:
        logger.error(f'rag_service // search runtime error: {ex}')
        return jsonify({'error': 'Search failed'}), 500
    except Exception as ex:
        logger.error(f'rag_service // search error: {ex}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for RAG service.
    """
    ready = RAG_INDEX is not None and bool(RAG_CHUNKS)
    payload = {
        'status': 'ok' if ready else 'not_ready',
        'index_ready': ready,
        'chunks': len(RAG_CHUNKS),
    }
    status_code = 200 if ready else 503
    logger.info(
        f'rag_service // health status={payload["status"]}, chunks={payload["chunks"]}'
    )
    return jsonify(payload), status_code


if __name__ == '__main__':
    app.run(
        host='localhost',
        port=config.tool_rag_port,
        debug=config.flask_debug,
    )
