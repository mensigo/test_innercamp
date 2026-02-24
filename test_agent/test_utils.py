import pytest
import requests

from src_agent import post_chat_completions, post_embeddings


class DummyResponse:
    def __init__(self, json_data: dict, status_code: int = 200, text: str = ''):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f'{self.status_code} error', response=self
            )


@pytest.mark.llm
def test_post_chat_completions_ok():
    """Call chat completions with a real request and expect content in answer."""
    payload = {
        'messages': [{'role': 'user', 'content': 'Hello from pytest, say hi'}],
        'max_tokens': 2,
    }
    result = post_chat_completions(payload)

    assert isinstance(result, dict)
    assert 'choices' in result
    assert result['choices'][0]['message']['content']


@pytest.mark.unit
def test_post_chat_completions_request_exception(monkeypatch: pytest.MonkeyPatch):
    """Handle network-related RequestException."""

    def fake_post(url: str, json: dict, **kwargs):
        raise requests.exceptions.Timeout('request timed out')

    monkeypatch.setattr('requests.post', fake_post)

    payload = {'messages': [{'role': 'user', 'content': 'Hello from pytest'}]}
    result = post_chat_completions(payload)

    assert isinstance(result, dict)
    assert 'error' in result
    assert 'request timed out' in result['error']


@pytest.mark.llm
def test_post_embeddings_ok():
    """Request embeddings with a real call and expect vector data."""
    payload = {'input': ['Hello from pytest']}
    result = post_embeddings(payload)

    assert isinstance(result, dict)
    assert 'data' in result
    assert result['data'][0]['embedding']


@pytest.mark.unit
def test_post_embeddings_http_error(monkeypatch: pytest.MonkeyPatch):
    """Handle embeddings HTTPError by returning error dict."""

    def fake_post(url: str, json: dict, **kwargs):
        return DummyResponse({}, status_code=502, text='bad gateway')

    monkeypatch.setattr('requests.post', fake_post)

    payload = {'input': ['Hello from pytest']}
    result = post_embeddings(payload)

    assert 'error' in result
    assert 'bad gateway' in result['error']
