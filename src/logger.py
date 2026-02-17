import sys

from loguru import logger


def stdout_only_filter(record: dict) -> bool:
    return record['level'].name == 'INFO'


logger.remove()  # remove default handler

logger.add(
    sys.stdout,
    # format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
    format='<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan> | {message}',
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
