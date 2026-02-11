from src import config, post_chat_completions


def classify_intent(user_input: str, **kwargs) -> bool:
    """
    Classify if user request is about Japanese poetry or haiku generation.
    """
    print('[classify_intent] start')

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
            {'role': 'user', 'content': user_input},
        ],
        'temperature': kwargs.get('temperature', config.freezing),
        'max_tokens': 1,
    }

    response = post_chat_completions(payload, verbose=kwargs.get('verbose', False))

    if 'error' in response:
        print('[classify_intent] LLM Error: {}'.format(response['error']))
        return False

    try:
        content = response['choices'][0]['message']['content'].strip().lower()
        return 'да' in content or 'yes' in content

    except (KeyError, IndexError) as e:
        print(f'[classify_intent] LLM Response Error: {e}')
        return False
