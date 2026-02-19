"""RAG tool client for poetry questions."""

import requests

from src import config

from .logger import logger


def _check_health(timeout: int = 10) -> bool:
    """
    Check RAG service health endpoint.
    """
    url = f'http://localhost:{config.tool_rag_port}/health'
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return bool(data.get('index_ready'))

    except Exception as ex:
        logger.error(f'rag_utils // Неожиданная ошибка: {ex}')
        return False


def _fetch_answer(
    question: str,
    top_k: int = 2,
    health_timeout: int = 5,
    search_timeout: int = 10,
) -> dict:
    """
    Fetch answer and chunk metadata from RAG service.
    """
    if not _check_health(health_timeout):
        return {'error': 'Health check failed'}

    payload = {'question': question, 'top_k': top_k}

    url = f'http://localhost:{config.tool_rag_port}/search'
    try:
        response = requests.post(url, json=payload, timeout=search_timeout)
        response.raise_for_status()
        return response.json()

    except Exception as ex:
        logger.error(f'rag_utils // Неожиданная ошибка - {ex}')
        return {'error': f'Unexpected error: {ex}'}


def answer_question(
    question: str,
    top_k: int = 2,
    health_timeout: int = 5,
    search_timeout: int = 10,
) -> dict:
    """
    Answer poetry question using RAG service.
    """
    response = _fetch_answer(question, top_k, health_timeout, search_timeout)

    if 'error' in response:
        return {'error': response['error']}

    return {
        'answer': response['answer'],
        'chunk_title_list': response['chunk_title_list'],
        'chunk_texts': response['chunk_texts'],
    }
