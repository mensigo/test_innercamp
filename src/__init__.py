from .config import config

if config.insigma:
    from .utils import post_chat_completions, post_embeddings
else:
    from .utils_openai import post_chat_completions, post_embeddings


__version__ = '0.1.0'

__all__ = ['config', 'post_chat_completions', 'post_embeddings']
