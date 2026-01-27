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
    assert value.get('object') == 'chat.completion'
    assert value.get('model')
    assert isinstance(value.get('created'), int)
    usage = value.get('usage')
    assert isinstance(usage, dict)
    assert usage.get('prompt_tokens') is not None
    assert usage.get('completion_tokens') is not None
    assert usage.get('total_tokens') is not None
    choices = value.get('choices')
    assert isinstance(choices, list) and choices, 'Choices missing from response'
    first_choice = choices[0]
    assert isinstance(first_choice, dict)
    assert first_choice.get('index') == 0
    assert first_choice.get('finish_reason')
    message = first_choice.get('message')
    assert isinstance(message, dict)
    assert message.get('role') == 'assistant'
    content = message.get('content')
    function_call = message.get('function_call')
    if function_call is not None:
        assert isinstance(function_call, dict)
        assert function_call.get('name')
        assert isinstance(function_call.get('arguments'), dict)
    else:
        assert isinstance(content, str) and content.strip(), 'Message content is empty'


def test_post_embeddings():
    """Request embeddings for a sample prompt."""
    payload = {'input': ['Hello from pytest']}
    result = post_embeddings(payload)
    assert isinstance(result, dict)
    assert result.get('object') == 'list'
    assert result.get('model')
    data = result.get('data')
    assert isinstance(data, list) and data, 'Expected data list in embeddings result'
    for index, entry in enumerate(data):
        assert isinstance(entry, dict)
        assert entry.get('object') == 'embedding'
        embedding = entry.get('embedding')
        assert isinstance(embedding, list), 'Embedding should be a list'
        assert entry.get('index') == index
        usage = entry.get('usage')
        assert isinstance(usage, dict)
        assert usage.get('prompt_tokens') is not None


def test_get_models():
    """Request the list of available models."""
    result = get_models()
    assert isinstance(result, dict)
    assert result.get('object') == 'list'
    data = result.get('data')
    assert isinstance(data, list) and data, 'Expected data list in models result'
    for entry in data:
        assert isinstance(entry, dict)
        assert entry.get('id'), 'Model entry missing id'
        assert entry.get('object') == 'model'
        assert entry.get('owned_by'), 'Model entry missing owner'
        assert entry.get('type'), 'Model entry missing type'
