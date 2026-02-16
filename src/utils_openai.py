"""OpenAI API wrapper functions using OpenRouter."""

import requests

from src import config, logger

DEFAULT_MODEL = 'openai/gpt-3.5-turbo'
DEFAULT_EMBEDDING_MODEL = 'openai/text-embedding-3-small'


def post_chat_completions(payload: dict, verbose: bool = False) -> dict:
    """
    Send chat completion request to OpenRouter.
    Uses openai/gpt-3.5-turbo model by default.
    """
    url = f'{config.openrouter_base_url}/chat/completions'

    if 'model' not in payload:
        payload['model'] = DEFAULT_MODEL

    headers = {
        'Authorization': f'Bearer {config.openrouter_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': config.openrouter_referer,
        'X-Title': config.openrouter_title,
    }

    try:
        if verbose:
            logger.debug('post/req: {}', payload)

        response = requests.post(url, json=payload, headers=headers, timeout=30)

        if verbose:
            logger.debug('post/ans: {} | {}', response, response.text)

        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as ex:
        logger.exception('post/error: {}', ex)
        text = ex.response.text if ex.response is not None else ''
        return {'error': f'{ex} {text}'.strip()}

    except requests.exceptions.RequestException as ex:
        logger.exception('post/error: {}', ex)
        return {'error': str(ex)}


def post_embeddings(payload: dict, verbose: bool = False) -> dict:
    """
    Send embeddings request to OpenRouter.
    Uses text-embedding-3-small model by default.
    """
    url = f'{config.openrouter_base_url}/embeddings'

    if 'model' not in payload:
        payload['model'] = DEFAULT_EMBEDDING_MODEL

    headers = {
        'Authorization': f'Bearer {config.openrouter_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': config.openrouter_referer,
        'X-Title': config.openrouter_title,
    }

    try:
        if verbose:
            logger.debug('post/req: {}', payload)

        response = requests.post(url, json=payload, headers=headers, timeout=30)

        if verbose:
            logger.debug('post/ans: {} | {}', response, response.text[:500])

        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as ex:
        logger.exception('post/error: {}', ex)
        text = ex.response.text if ex.response is not None else ''
        return {'error': f'{ex} {text}'.strip()}

    except requests.exceptions.RequestException as ex:
        logger.exception('post/error: {}', ex)
        return {'error': str(ex)}
