"""Parameter clarification logic."""


def generate_clarification_prompt(tool_name: str, missing_param: str) -> str:
    """
    Generate clarification prompt for missing parameter.
    """
    if tool_name == 'generate_haiku' and missing_param == 'theme':
        return 'На какую тему мне составить хайку? (просьба переформулировать, если вы ее указали ранее)'

    if tool_name == 'rag_search' and missing_param == 'question':
        return 'Уточните ваш вопрос о японской поэзии'

    return f'Уточните параметр: {missing_param}'


def extract_param_from_clarification(
    user_input: str, param_name: str, tool_name: str
) -> str | None:
    """
    Extract parameter from user clarification response.
    Returns extracted parameter or None if extraction failed.
    """
    user_input = user_input.strip()

    if not user_input:
        return None

    # Игнорируем слишком короткие или неинформативные ответы
    lowered = user_input.lower()
    ignore_phrases = {
        'не знаю',
        'незнаю',
        'хз',
        'что-нибудь',
        'что угодно',
        'любая',
        'любую',
        'любой',
    }
    if lowered in ignore_phrases:
        return None

    if param_name == 'theme':
        # Для темы: проверяем длину и базовую валидность
        if len(user_input) > 20:
            return None
        # Если в ответе есть "про", извлекаем тему после него
        if 'про ' in lowered:
            idx = lowered.index('про ') + 4
            theme = user_input[idx:].strip()
            if theme and len(theme) <= 20:
                return theme
        return user_input

    if param_name == 'question':
        # Для вопроса: принимаем любой непустой ввод
        if len(user_input) < 3:
            return None
        return user_input

    return None
