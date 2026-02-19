"""Tool call validation logic."""

from ..logger import logger

MAX_QUESTION_LEN = 30
MAX_THEME_LEN = 20


def validate_tool_call(tool_name: str, tool_args: dict) -> tuple[bool, dict]:
    """
    Validate tool arguments before calling tool.
    Returns:
        True, {"message": str} - if valid
        False, {"message": str, "param": str, "reason": str} - if param invalid/missing or tool unknown
    """
    if tool_name == 'rag_search':
        if 'question' not in tool_args:
            logger.warning('validate_tool_call // Missing Param: rag_search::question')
            return (
                False,
                {
                    'message': 'Не совсем понял вопрос. Просьба переформулировать.',
                    'param': 'question',
                    'reason': 'missing',
                },
            )

        question = str(tool_args.get('question', '')).strip()
        if not question:
            logger.warning('validate_tool_call // Empty Param: rag_search::question')
            return (
                False,
                {
                    'message': 'Не совсем понял вопрос. Просьба переформулировать.',
                    'param': 'question',
                    'reason': 'empty',
                },
            )

        if len(question) > MAX_QUESTION_LEN:
            logger.warning('validate_tool_call // Too Long Param: rag_search::question')
            return (
                False,
                {
                    'message': 'Вопрос слишком длинный. Просьба сформулировать более кратко.',
                    'param': 'question',
                    'reason': 'long',
                },
            )

        logger.debug('validate_tool_call // Validation OK')
        return (
            True,
            {
                'message': (
                    f'Инструмент rag_search проверен и готов к вызову. Запрос: {question}'
                )
            },
        )

    if tool_name == 'generate_haiku':
        if 'theme' not in tool_args:
            logger.warning('validate_tool_call // Missing Param: generate_haiku::theme')
            return (
                False,
                {
                    'message': 'Не совсем понял вопрос. Просьба переформулировать.',
                    'param': 'theme',
                    'reason': 'missing',
                },
            )

        theme = str(tool_args['theme']).strip()
        if not theme:
            logger.warning('validate_tool_call // Empty Param: generate_haiku::theme')
            return (
                False,
                {
                    'message': 'Не совсем понял вопрос. Просьба переформулировать.',
                    'param': 'theme',
                    'reason': 'empty',
                },
            )

        if len(theme) > MAX_THEME_LEN:
            logger.warning(
                'validate_tool_call // Too Long Param: generate_haiku::theme'
            )
            return (
                False,
                {
                    'message': 'Тема слишком длинная. Просьба сформулировать более кратко.',
                    'param': 'theme',
                    'reason': 'long',
                },
            )

        logger.debug('validate_tool_call // Validation OK')
        return (
            True,
            {
                'message': (
                    f'Инструмент generate_haiku проверен и готов к вызову. Тема: {theme}'
                )
            },
        )

    logger.warning(f'validate_tool_call // Unknown tool: {tool_name}')
    return (
        False,
        {
            'message': 'Не удалось провалидировать запрос. Просьба переформулировать.',
            'param': tool_name,
            'reason': 'unknown_tool',
        },
    )
