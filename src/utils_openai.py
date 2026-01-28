"""OpenAI API wrapper functions using OpenRouter."""

import requests

from src import config

DEFAULT_MODEL = 'openai/gpt-3.5-turbo'
DEFAULT_EMBEDDING_MODEL = 'openai/text-embedding-3-small'


def post_chat_completions(payload: dict) -> dict:
    """
    Send chat completion request to OpenRouter.
    Uses openai/gpt-3.5-turbo model by default.
    """
    url = f'{config.openrouter_base_url}/chat/completions'

    # Set default model if not provided in payload
    if 'model' not in payload:
        payload['model'] = DEFAULT_MODEL

    headers = {
        'Authorization': f'Bearer {config.openrouter_key}',
        'Content-Type': 'application/json',
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def post_embeddings(payload: dict) -> dict:
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
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}
