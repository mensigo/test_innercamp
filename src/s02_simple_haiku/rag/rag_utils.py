"""RAG tool client for poetry questions."""

import requests

from src import config
from .logger import logger


def check_health() -> bool:
    """
    Check RAG service health endpoint.
    """
    url = f'http://localhost:{config.tool_rag_port}/health'
    try:
        response = requests.get(url, timeout=4)
        response.raise_for_status()
        data = response.json()
        return bool(data.get('index_ready'))

    except Exception as ex:
        logger.error(f'rag_utils // Неожиданная ошибка - {ex}')
        return False


def fetch_answer(question: str, top_k: int = 2) -> dict:
    """
    Fetch answer and chunk metadata from RAG service.
    """
    if not check_health():
        return {}

    payload = {'question': question, 'top_k': top_k}

    url = f'http://localhost:{config.tool_rag_port}/search'
    try:
        response = requests.post(url, json=payload, timeout=8)
        response.raise_for_status()
        return response.json()

    except Exception as ex:
        logger.critical(f'rag_utils // Неожиданная ошибка - {ex}')
        return {}


def answer_question(question: str) -> dict:
    """
    Answer poetry question using RAG service.
    """
    response = fetch_answer(question)
    if not isinstance(response, dict) or not response:
        return {
            'answer': 'Не удалось получить ответ от RAG сервиса.',
            'chunk_title_list': [],
            'chunk_texts': [],
        }
    if 'error' in response:
        return {
            'answer': str(response.get('error')),
            'chunk_title_list': [],
            'chunk_texts': [],
        }
    chunk_titles = response.get('chunk_title_list', [])
    chunk_texts = response.get('chunk_texts', [])
    if not isinstance(chunk_titles, list):
        chunk_titles = []
    if not isinstance(chunk_texts, list):
        chunk_texts = []
    return {
        'answer': str(response.get('answer', '')),
        'chunk_title_list': chunk_titles,
        'chunk_texts': chunk_texts,
    }
