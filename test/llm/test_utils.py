from __future__ import annotations

import pytest

from src.utils import post_chat_completions, post_embeddings

pytestmark = [pytest.mark.llm]


def test_post_chat_completions_happy_path():
    payload = {
        'messages': [
            {'role': 'user', 'content': 'Ответь одним словом: привет'},
        ],
        'max_tokens': 5,
    }
    result = post_chat_completions(payload)

    assert isinstance(result, dict)
    assert result.get('choices')

    first_choice = result['choices'][0]['message']['content']
    assert isinstance(first_choice, str)
    assert first_choice.strip()


def test_post_embeddings_happy_path():
    payload = {'input': 'машинное обучение'}

    result = post_embeddings(payload)

    assert isinstance(result, dict)
    assert result.get('data')

    first_item = result['data'][0]['embedding']
    assert isinstance(first_item, list)
    assert all(isinstance(value, float) for value in first_item)
