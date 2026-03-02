"""GigaChat API wrapper functions."""


def post_chat_completions(payload: dict, **kwargs) -> dict:
    """
    Generate model response based on messages.
    Sends POST request to /chat/completions endpoint.
    """
    # raise NotImplementedError('Not implemented')
    from src_example import post_chat_completions

    return post_chat_completions(payload, **kwargs)


def post_embeddings(payload: dict, **kwargs) -> dict:
    """
    Create vector embeddings for text.
    Sends POST request to /embeddings endpoint.
    """
    # raise NotImplementedError('Not implemented')
    from src_example import post_embeddings

    return post_embeddings(payload, **kwargs)
