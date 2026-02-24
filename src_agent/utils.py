"""GigaChat API wrapper functions."""

from src.utils import post_chat_completions as post_chat_completions_impl
from src.utils import post_embeddings as post_embeddings_impl


def post_chat_completions(payload: dict, verbose: bool = False) -> dict:
    """
    Generate model response based on messages.
    Sends POST request to /chat/completions endpoint.
    """
    return post_chat_completions_impl(payload, verbose)


def post_embeddings(payload: dict, verbose: bool = False) -> dict:
    """
    Create vector embeddings for text.
    Sends POST request to /embeddings endpoint.
    """
    return post_embeddings_impl(payload, verbose)
