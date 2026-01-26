import os
import pprint
from typing import Any

import pytest

from src.utils_openai import post_chat_completions


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
