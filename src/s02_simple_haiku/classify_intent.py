from src import config, logger, post_chat_completions


def classify_intent(message_history: list[dict], **kwargs) -> int:
    """
    Classify if user request is about Japanese poetry or haiku generation.
    Returns:
        0 - relevant request
        1 - irrelevant request
        2 - error caught
    """
    system_prompt = """Ты классификатор запросов.
Определи, относится ли запрос к японской поэзии или генерации хайку.

Примеры запросов ДЛЯ НАШЕГО АГЕНТА (отвечай "да"):
- "напиши хайку о море"
- "сгенерируй хокку"
- "что такое хайку?"
- "кто такие рюкюсцы?"
- "сколько канси в Манъёсю?"

Примеры запросов НЕ ДЛЯ НАШЕГО АГЕНТА (отвечай "нет"):
- "напиши стишок"
- "расскажи анекдот"
- "погода завтра"
- "что там с гренландией?"

Отвечай ТОЛЬКО одним словом: "да" или "нет"."""

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            *message_history,
        ],
        'temperature': kwargs.get('temperature', config.freezing),
        'max_tokens': 1,
    }

    response = post_chat_completions(payload, verbose=kwargs.get('verbose', False))

    if 'error' in response:
        logger.critical('classify_intent // LLM Error: {}'.format(response['error']))
        return 2

    try:
        content = response['choices'][0]['message']['content'].strip().lower()
        return 0 if ('да' in content or 'yes' in content) else 1

    except (KeyError, IndexError) as e:
        print(f'classify_intent // LLM Response Parse Error: {e}')
        return 2
