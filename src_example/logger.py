import os
import sys

from loguru import logger


def stdout_only_filter(record: dict) -> bool:
    return record['level'].name == 'INFO'


def debug_only_filter(record: dict) -> bool:
    return record['level'].name == 'DEBUG'


logger.remove()  # remove default handler

logger.add(
    sys.stdout,
    format='<dim>{message}</dim>',
    level='INFO',
    colorize=True,
    filter=stdout_only_filter,
)
logger.add(
    'agent.log',
    format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
    level='DEBUG',
    rotation='5 MB',
    retention='1 days',
)
if os.environ.get('LOG_DEBUG_STDOUT'):
    logger.add(
        sys.stdout,
        format='<dim>{level} | {message}</dim>',
        level='DEBUG',
        colorize=True,
        filter=debug_only_filter,
    )
