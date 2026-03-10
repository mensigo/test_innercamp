"""OpenAI API wrapper functions using OpenRouter."""

import time

import requests

from .config import config
from .logger import logger

CHAT_COMPLETIONS_RETRY_ATTEMPTS = 5
CHAT_COMPLETIONS_RETRY_DELAY_SECONDS = 8
UNSUPPORTED_REGION_ERROR_MARKER = 'unsupported_country_region_territory'


def post_chat_completions(payload: dict, verbose: bool = False) -> dict:
    """
    Send chat completion request to OpenRouter.
    Uses openai/gpt-3.5-turbo model by default.
    """
    url = f'{config.openrouter_base_url}/chat/completions'

    if 'model' not in payload:
        payload['model'] = config.default_model

    headers = {
        'Authorization': f'Bearer {config.openrouter_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': config.openrouter_referer,
        'X-Title': config.openrouter_title,
    }

    for attempt in range(1, CHAT_COMPLETIONS_RETRY_ATTEMPTS + 1):
        try:
            if verbose:
                logger.debug('post/req: {}', payload)

            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if verbose:
                logger.debug('post/ans: {} | {}', response, response.text.strip())

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as ex:
            text = ex.response.text if ex.response is not None else ''
            status_code = ex.response.status_code if ex.response is not None else None
            should_retry = (
                status_code == 403
                and UNSUPPORTED_REGION_ERROR_MARKER in text
                and attempt < CHAT_COMPLETIONS_RETRY_ATTEMPTS
            )
            if should_retry:
                logger.warning(
                    'post/retry: attempt {}/{} after {}s due to region restriction',
                    attempt,
                    CHAT_COMPLETIONS_RETRY_ATTEMPTS,
                    CHAT_COMPLETIONS_RETRY_DELAY_SECONDS,
                )
                time.sleep(CHAT_COMPLETIONS_RETRY_DELAY_SECONDS)
                continue

            logger.error('post/error: {}', ex)
            return {'error': f'{ex} {text}'.strip()}

        except requests.exceptions.RequestException as ex:
            logger.error('post/error: {}', ex)
            return {'error': str(ex)}


def post_embeddings(payload: dict, verbose: bool = False) -> dict:
    """
    Send embeddings request to OpenRouter.
    Uses text-embedding-3-small model by default.
    """
    url = f'{config.openrouter_base_url}/embeddings'

    if 'model' not in payload:
        payload['model'] = config.default_embedding_model

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
        logger.error('post/error: {}', ex)
        text = ex.response.text if ex.response is not None else ''
        return {'error': f'{ex} {text}'.strip()}

    except requests.exceptions.RequestException as ex:
        logger.error('post/error: {}', ex)
        return {'error': str(ex)}
