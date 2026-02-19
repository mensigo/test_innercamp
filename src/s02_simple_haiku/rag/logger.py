"""Logger for RAG build/search tools."""

import sys
from pathlib import Path

from loguru import logger

LOG_PATH = Path(__file__).resolve().parent / 'tool_rag_search.log'

# Reset default handlers
logger.remove()

# Stdout sink
STDOUT_SINK_ID = logger.add(
    sys.stdout,
    format='<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}',
    level='INFO',
    colorize=True,
)

# File sink
FILE_SINK_ID = logger.add(
    LOG_PATH,
    format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
    level='DEBUG',
    rotation='5 MB',
    retention='3 days',
)
