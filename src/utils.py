"""GigaChat API wrapper functions."""


def get_chat_completions(payload: dict, **kwargs) -> dict:
    """
    Generate model response based on messages.
    """
    # raise NotImplementedError('Not implemented')
    from src_example import post_chat_completions

    return post_chat_completions(payload, **kwargs)


def get_embeddings(payload: dict, **kwargs) -> list[float]:
    """
    Create vector embeddings for input text.

    Example input: {'input': 'машинное обучение'}
    Example output: [0.935546875, -0.092529296]
    """
    # raise NotImplementedError('Not implemented')
    from src_example import post_embeddings

    assert 'input' in payload, 'input is required'
    assert isinstance(payload['input'], str), 'input must be a string'

    response = post_embeddings(payload, **kwargs)
    return response['data'][0]['embedding']
