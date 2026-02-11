"""Tool call validation logic."""


def detect_theme_change(user_input: str) -> str | None:
    """
    Detect theme change intent and extract new theme if provided.
    """
    print('[detect_theme_change] start')

    lowered = user_input.lower()
    triggers = [
        'сменить тему на ',
        'смени тему на ',
        'поменяй тему на ',
        'измени тему на ',
        'изменить тему на ',
    ]

    for trigger in triggers:
        if trigger in lowered:
            start = lowered.index(trigger) + len(trigger)
            return user_input[start:].strip()

    if 'сменить тему' in lowered or 'другая тема' in lowered:
        return ''

    return None


def validate_tool_call(
    tool_name: str, tool_args: dict, user_input: str
) -> tuple[bool, str | None]:
    """
    Validate tool arguments before calling tool.
    Returns (is_valid, missing_param_name).
    """
    print('[validate_tool_call] start')

    if tool_name == 'rag_search':
        question = str(tool_args.get('question', '')).strip()
        if not question:
            return False, 'question'
        return True, None

    if tool_name == 'generate_haiku':
        theme = str(tool_args.get('theme', '')).strip()
        theme_change = detect_theme_change(user_input)
        if theme_change is not None:
            if theme_change:
                print(
                    f'[validate_tool_call | theme_change] Тема изменена на: {theme_change}'
                )
            else:
                print(
                    '[validate_tool_call | theme_change] Укажите новую тему после команды смены темы.'
                )
            print('[validate_tool_call | theme_change] Сформулируй запрос заново.\n')
            return False, None

        if not theme:
            return False, 'theme'

        if len(theme) > 20:
            print(
                '[validate_tool_call | validation_error] Тема слишком длинная. Сократи до 1-2 слов.'
            )
            return False, None

        return True, None

    print('[validate_tool_call | error] Неизвестный инструмент.')
    return False, None
