from src import config, logger, post_chat_completions


def classify_intent(message_history: list[dict] | str, **kwargs) -> int:
    """
    Classify if user request is about Japanese poetry or haiku generation.
    Accepts raw user text or prepared message history.
    Returns:
        0 - relevant request
        1 - irrelevant request
        2 - request error
        3 - parsing error
    """
    if isinstance(message_history, str):
        message_history = [{'role': 'user', 'content': message_history}]

    system_prompt = """Ты классификатор запросов.
Твоя задача - определить, относится ли запрос к японской поэзии или генерации хайку.
Перед тобой диалог с пользователем. Определи последний запрос (intent) пользователя, затем ответь "да" или "нет".

# Примеры запросов ДЛЯ НАШЕГО АГЕНТА (отвечай "да")

## Одна реплика
- "напиши хайку о море"
- "сгенерируй хокку"
- "что такое хайку?"
- "кто такие рюкюсцы?"
- "сколько канси в Манъёсю?"

## Несколько реплик

1.
Пользователь: напиши что-нибудь
Ассистент: Запрос не связан с функционалом агента.
Пользователь: хайку

2.
Пользователь: write smth
Ассистент: Запрос не связан с функционалом агента.
Пользователь: haiku

# Примеры запросов НЕ ДЛЯ НАШЕГО АГЕНТА (отвечай "нет")

## Одна реплика
- "напиши стишок"
- "расскажи анекдот"
- "погода завтра"
- "что там с гренландией?"

## Несколько реплик

1.
Пользователь: стишок про канарейку
Ассистент: Запрос не связан с функционалом агента.
Пользователь: ну пожалуйста

2.
Пользователь: погода в Агробе
Ассистент: Запрос не связан с функционалом агента.
Пользователь: а в Судане?

---
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
        if 'да' in content or 'yes' in content:
            logger.debug('classify_intent // Relevant query')
            return 0

        logger.warning('classify_intent // Irrelevant query')
        return 1

    except (KeyError, IndexError) as ex:
        logger.opt(exception=True).critical(
            f'classify_intent // LLM Response Parse Error: {ex}'
        )
        return 3
