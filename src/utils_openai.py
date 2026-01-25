"""OpenAI API wrapper functions using OpenRouter."""

import os
import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Global configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
DEFAULT_MODEL = 'openai/gpt-3.5-turbo'


def post_chat_completions(payload: dict) -> dict:
    """
    Send chat completion request to OpenRouter.
    Uses openai/gpt-3.5-turbo model by default.
    """
    url = f'{OPENROUTER_BASE_URL}/chat/completions'

    # Set default model if not provided in payload
    if 'model' not in payload:
        payload['model'] = DEFAULT_MODEL

    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}
