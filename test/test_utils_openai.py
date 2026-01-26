import os
import pprint

import pytest

from src.utils_openai import post_chat_completions


INSIGMA = os.getenv('INSIGMA', 'false').lower() == 'true'
pytestmark = pytest.mark.skipif(
    INSIGMA, reason='Skip OpenAI test whenever INSIGMA is enabled'
)


def test_chat_completions() -> None:
    """Test chat completions with OpenRouter."""
    payload = {
        'messages': [{'role': 'user', 'content': 'Hello, how are you?'}],
        'max_tokens': 10,
    }

    result = post_chat_completions(payload)
    pprint.pprint(result)
