import pprint
from src.utils_openai import post_chat_completions


def test_chat_completions():
    """Test chat completions with OpenRouter."""
    payload = {
        'messages': [{'role': 'user', 'content': 'Hello, how are you?'}],
        'max_tokens': 10,
    }

    result = post_chat_completions(payload)
    pprint.pprint(result)


if __name__ == '__main__':
    test_chat_completions()
