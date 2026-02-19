"""RAG tool client for poetry questions."""

import requests

from src import config, logger


def check_health(timeout: int = 10) -> bool:
    """
    Check RAG service health endpoint.
    """
    url = f'http://localhost:{config.tool_rag_port}/health'
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        logger.debug(f'check_health // Recieved data: {data}')
        return data['status'] == 'ok'

    except Exception as ex:
        logger.error(f'check_health // Unexpected error: {ex}')
        return False


def answer_question(
    question: str,
    top_k: int = 2,
    health_timeout: int = 5,
    search_timeout: int = 10,
) -> dict:
    """
    Answer poetry question using RAG service.
    """
    if not check_health(health_timeout):
        logger.error('answer_question // Health check failed')
        return {'error': 'Health check failed'}

    try:
        url = f'http://localhost:{config.tool_rag_port}/search'
        payload = {'question': question, 'top_k': top_k}
        response = requests.post(url, json=payload, timeout=search_timeout)
        response.raise_for_status()
        data = response.json()
        logger.debug(f'answer_question // Recieved data: {data}')

        if 'error' in data:
            logger.error('answer_question // Search error: {}'.format(data['error']))
            return {'error': 'Search error: {}'.format(data['error'])}

        return {
            'answer': data['answer'],
            'chunk_title_list': data['chunk_title_list'],
            'chunk_texts': data['chunk_texts'],
        }

    except Exception as ex:
        logger.error(f'answer_question // Unexpected error: {ex}')
        return {'error': f'Unexpected error: {ex}'}
