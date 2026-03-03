"""RAG retrieval and answer extraction helpers for the example agent."""

from __future__ import annotations

from src.api import vector_search

from .config import config
from .logger import logger
from .utils import post_chat_completions


def _read_response_text(response: dict) -> str:
    """Extract assistant content text from chat completions response."""
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices:
        return ''

    message = choices[0].get('message')
    if not isinstance(message, dict):
        return ''

    content = message.get('content')
    if not isinstance(content, str):
        return ''
    return content.strip()


def _extract_answer_from_chunks(user_query: str, chunks: list[str]) -> str:
    """Ask LLM to extract exact answer from retrieved chunks."""
    if not chunks:
        return ''

    chunks_block = '\n\n'.join(
        [f'Фрагмент {idx}:\n{chunk}' for idx, chunk in enumerate(chunks, start=1)]
    )
    payload = {
        'messages': [
            {
                'role': 'system',
                'content': (
                    'Ты извлекаешь ответ из фрагментов текста.\n'
                    'Правила:\n'
                    '1) Используй только предоставленные фрагменты.\n'
                    '2) Верни ответ ровно как найдено в тексте, без перефразирования.\n'
                    '3) Не добавляй пояснений, вводных слов и кавычек.\n'
                    '4) Если точного ответа нет, верни пустую строку.'
                ),
            },
            {
                'role': 'user',
                'content': f'Вопрос:\n{user_query}\n\nФрагменты:\n{chunks_block}',
            },
        ],
        'temperature': config.freezing,
        'max_tokens': 200,
    }
    response = post_chat_completions(payload, verbose=config.debug)
    if 'error' in response:
        logger.warning(f'agent // rag extract error: {response["error"]}')
        return ''
    return _read_response_text(response)


def answer_with_rag(user_query: str, retrieval_query: str, top_k: int = 2) -> str:
    """Retrieve top-k chunks and extract answer with LLM."""
    rag_result = vector_search(query=retrieval_query, k=top_k)
    raw_chunks = rag_result.get('chunks')
    if not isinstance(raw_chunks, list):
        return ''
    chunks = [chunk for chunk in raw_chunks if isinstance(chunk, str) and chunk.strip()]
    return _extract_answer_from_chunks(user_query=user_query, chunks=chunks)
