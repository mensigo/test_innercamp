import os
from typing import Any

import pytest

from src.utils import get_models, post_chat_completions


INSIGMA = os.getenv('INSIGMA', 'false').lower() == 'true'
pytestmark = pytest.mark.skipif(
    not INSIGMA, reason='Run Gigachat helpers only when INSIGMA is true'
)


def _assert_response_has_data(response: dict[str, Any]) -> None:
    """Ensure the helper returns at least one field in the response."""
    assert isinstance(response, dict)
    assert response, 'Expected non-empty response from Gigachat helper'


def test_post_chat_completions() -> None:
    """Call chat completions with a short prompt."""
    payload = {
        'messages': [{'role': 'user', 'content': 'Hello from pytest'}],
        'max_tokens': 4,
    }

    result = post_chat_completions(payload)
    _assert_response_has_data(result)


def test_get_models() -> None:
    """Request the list of available models."""
    result = get_models()
    _assert_response_has_data(result)
