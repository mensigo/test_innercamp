"""Client for haiku generation service."""

import requests

from src import config, logger


def check_health(timeout: int = 4) -> bool:
    """
    Check haiku service health endpoint.
    """
    url = f'http://localhost:{config.tool_haiku_port}/health'
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        logger.debug(f'check_health // Received data: {data}')
        return data.get('status') == 'ok'
    except Exception as ex:
        logger.error(f'check_health // Unexpected error: {ex}')
        return False


def generate_haiku(
    theme: str, health_timeout: int = 5, generate_timeout: int = 10
) -> dict:
    """
    Generate haiku using haiku service.
    """
    if not check_health(health_timeout):
        logger.error('generate_haiku // Health check failed')
        return {'error': 'Health check failed'}

    url = f'http://localhost:{config.tool_haiku_port}/generate_haiku'
    payload = {'theme': theme}

    try:
        response = requests.post(url, json=payload, timeout=generate_timeout)
        response.raise_for_status()
        data = response.json()
        logger.debug(f'generate_haiku // Received data: {data}')

        if 'error' in data:
            logger.error('generate_haiku // Generation error: {}'.format(data['error']))
            return {'error': 'Generation error: {}'.format(data['error'])}

        return {
            'haiku_text': data['haiku_text'],
            'syllables_per_line': data['syllables_per_line'],
            'total_words': data['total_words'],
            'theme': data['theme'],
        }

    except Exception as ex:
        logger.error(f'generate_haiku // Unexpected error: {ex}')
        return {'error': f'Unexpected error: {ex}'}
