import os

import pytest

from src.utils import get_models, post_chat_completions, post_embeddings


INSIGMA = os.getenv('INSIGMA', 'false').lower() == 'true'
pytestmark = pytest.mark.skipif(
    not INSIGMA, reason='Run Gigachat helpers only when INSIGMA is true'
)


def test_post_chat_completions():
    """Call chat completions with a short prompt."""
    payload = {
        'messages': [{'role': 'user', 'content': 'Hello from pytest'}],
        'max_tokens': 4,
    }
    result = post_chat_completions(payload)
    assert isinstance(result, dict)
    assert result, 'Expected non-empty response from Gigachat helper'
    assert 'value' in result, 'Expected Gigachat response to include value'
    value = result['value']
    assert isinstance(value, dict)
    choices = value.get('choices')
    assert isinstance(choices, list) and choices, 'Choices missing from response'
    first_choice = choices[0]
    message = first_choice.get('message')
    assert isinstance(message, dict)
    content = message.get('content')
    assert isinstance(content, str) and content.strip(), 'Message content is empty'


def test_post_embeddings():
    """Request embeddings for a sample prompt."""
    payload = {'input': ['Hello from pytest']}
    result = post_embeddings(payload)
    assert isinstance(result, dict)
    assert result.get('object') == 'list'
    assert 'model' in result
    data = result.get('data')
    assert isinstance(data, list) and data, 'Expected data list in embeddings result'
    first = data[0]
    assert isinstance(first, dict)
    assert first.get('object') == 'embedding'
    embedding = first.get('embedding')
    assert isinstance(embedding, list), 'Embedding should be a list'
    assert first.get('index') == 0
    usage = first.get('usage')
    assert isinstance(usage, dict)
    assert usage.get('prompt_tokens') is not None


def test_get_models():
    """Request the list of available models."""
    result = get_models()
    assert isinstance(result, dict)
    assert result.get('object') == 'list'
    data = result.get('data')
    assert isinstance(data, list) and data, 'Expected data list in models result'
    first = data[0]
    assert isinstance(first, dict)
    assert first.get('id'), 'Model entry missing id'
    assert first.get('object') == 'model'
    assert first.get('owned_by'), 'Model entry missing owner'
