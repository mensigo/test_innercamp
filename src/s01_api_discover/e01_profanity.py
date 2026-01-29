"""Helpers that toggle the Gigachat profanity check flag."""

from src.utils import post_chat_completions


def _call_with_profanity(payload: dict, enable_profanity: bool, verbose: bool) -> dict:
    """Forward the payload with the desired profanity flag."""
    request_payload = dict(payload)
    request_payload['profanity_check'] = enable_profanity
    return post_chat_completions(request_payload, verbose)


def ask_with_profanity_check(payload: dict, verbose: bool = False) -> dict:
    """Invoke Gigachat with profanity filtering enforced."""
    return _call_with_profanity(payload, True, verbose)


def ask_without_profanity_check(payload: dict, verbose: bool = False) -> dict:
    """Invoke Gigachat with profanity filtering disabled."""
    return _call_with_profanity(payload, False, verbose)
