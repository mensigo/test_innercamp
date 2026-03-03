from __future__ import annotations

import pytest

from src.utils import get_chat_completions, get_embeddings

pytestmark = [pytest.mark.llm]


def test_get_chat_completions_happy_path():
    payload = {
        'messages': [
            {'role': 'user', 'content': 'Ответь одним словом: привет'},
        ],
        'max_tokens': 5,
    }
    result = get_chat_completions(payload)

    answer = result['choices'][0]['message']['content'].strip()
    assert answer


def test_get_embeddings_happy_path():
    payload = {'input': 'машинное обучение'}

    embedding = get_embeddings(payload)

    assert isinstance(embedding, list)
    assert embedding
    assert all(isinstance(value, float) for value in embedding)
