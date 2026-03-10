import os

import pytest

from src.utils_openai import post_chat_completions, post_embeddings

INSIGMA = os.getenv('INSIGMA', 'false').lower() == 'true'
pytestmark = pytest.mark.skipif(
    INSIGMA, reason='Skip OpenAI test whenever INSIGMA is enabled'
)


def test_chat_completions():
    """Test chat completions with OpenRouter."""
    payload = {
        'messages': [{'role': 'user', 'content': 'Hello, how are you?'}],
        'max_tokens': 10,
    }
    result = post_chat_completions(payload)
    assert isinstance(result, dict)
    assert result.get('choices'), 'Expected choices in OpenRouter result'
    choices = result['choices']
    assert isinstance(choices, list) and choices
    first_choice = choices[0]
    message = first_choice.get('message')
    assert isinstance(message, dict)
    content = message.get('content')
    assert isinstance(content, str) and content.strip(), 'Message content is empty'


def test_post_embeddings():
    """Request embeddings via OpenRouter."""
    payload = {'input': ['Hello from pytest']}
    result = post_embeddings(payload)
    assert isinstance(result, dict)
    assert result.get('object') == 'list'
    assert result.get('model')
    data = result.get('data')
    assert isinstance(data, list) and data
    first = data[0]
    assert isinstance(first, dict)
    assert first.get('object') == 'embedding'
    embedding = first.get('embedding')
    assert isinstance(embedding, list), 'Embedding should be a list'
    assert first.get('index') == 0
    usage = first.get('usage')
    assert isinstance(usage, dict)
    assert usage.get('prompt_tokens') is not None
    assert usage.get('total_tokens') is not None
