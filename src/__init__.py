from .config import config
from .logger import logger

if config.insigma:
    from .utils import post_chat_completions, post_embeddings
else:
    from .utils_openai import post_chat_completions, post_embeddings


__version__ = '0.1.0'

__all__ = ['config', 'logger', 'post_chat_completions', 'post_embeddings']
