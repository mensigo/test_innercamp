"""Tool call validation logic."""


def validate_tool_call(
    tool_name: str, tool_args: dict
) -> tuple[bool, tuple[str, str] | None]:
    """
    Validate tool arguments before calling tool.
    Returns:
        True, None - if valid
        False, (param_name, reason) - if param is invalid/missing
        False, None - if unknown tool
    """
    if tool_name == 'rag_search':
        if 'question' not in tool_args:
            return False, ('question', 'missing')

        if not str(tool_args['question']).strip():
            return False, ('question', 'empty')

        return True, None

    if tool_name == 'generate_haiku':
        if 'theme' not in tool_args:
            return False, ('theme', 'missing')

        theme = str(tool_args['theme']).strip()
        if not theme:
            return False, ('theme', 'empty')

        if len(theme) > 20:
            return False, ('theme', 'long')

        return True, None

    return False, None
