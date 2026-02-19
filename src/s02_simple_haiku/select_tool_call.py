import json

from src import config, logger, post_chat_completions


def select_tool_call(
    message_history: list[dict], **kwargs
) -> tuple[int, tuple[str, dict] | None]:
    """
    Select tool via function calling for the given user input.
    Returns:
        0,(name,args) - tool selected
        1,None - no tool selected
        2,None - request error
        3,None - parsing error
    """
    system_prompt = """
Твоя задача - определить, какой инструмент (tool) нужно вызвать в ответ на запрос пользователя.

Перед тобой диалог с пользователем. Определи последний запрос (intent) пользователя, затем подумай:
- если пользователь спрашивает о японской поэзии, вызови rag_search;
- если пользователь просит сгенерировать хайку/хокку (даже без указания темы), вызови generate_haiku.
Вызывай ровно один инструмент.

ВАЖНО:
- Если тема хайку не указана, все равно вызывай generate_haiku({}).
- Если это вопрос о японской поэзии, перефразируй его в более формальный вид (это и будет question).
"""

    if not config.insigma:
        system_prompt += """

## Примеры взаимодействия (generate_haiku):

Пользователь: Создай хайку о зимнем утре
Ты: generate_haiku({"theme": "Зимнее утро"})

Пользователь: Просто хайку
Ты: generate_haiku({})

Пользователь: пиши хокку мне
Ты: generate_haiku({})


## Примеры взаимодействия (rag_search):

Пользователь: Когда был написан манъесю
Ты: rag_search({"question": "Дата написания Манъёсю"})

Пользователь: Есть ли в старояпонском священные цифры?
Ты: rag_search({"question": "Священные цифры в старояпонском языке"})
"""

    # tool desctiption

    tools_giga = [
        {
            'name': 'rag_search',
            'description': 'Поиск ответа на вопрос о японской поэзии',
            'parameters': {
                'type': 'object',
                'properties': {
                    'question': {
                        'type': 'string',
                        'description': 'Вопрос пользователя о японской поэзии',
                    },
                },
                'required': [],
            },
            'few_shot_examples': [
                {
                    'request': 'Есть ли в старояпонском священные цифры?',
                    'params': {'question': 'Священные цифры в старояпонском языке?'},
                }
            ],
            'return_parameters': {
                'properties': {
                    'answer': {
                        'type': 'string',
                        'description': 'Подробный ответ (например, исторический факт или число)',
                    },
                    'chunk_title_list': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Заголовки статей, например ["Японская поэзия", "Старояпонский язык", "Кайфусо"]',
                    },
                    'chunk_texts': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Тексты найденных фрагментов из источников',
                    },
                }
            },
        },
        {
            'name': 'generate_haiku',
            'description': 'Генерация хайку по теме',
            'parameters': {
                'type': 'object',
                'properties': {
                    'theme': {
                        'type': 'string',
                        'description': 'Тема хайку одним-двумя словами',
                    },
                },
                'required': [],
            },
            'few_shot_examples': [
                {
                    'request': 'Напиши хайку о бурном море',
                    'params': {'theme': 'Бурное море'},
                },
            ],
            'return_parameters': {
                'properties': {
                    'haiku_text': {
                        'type': 'string',
                        'description': 'Сгенерированное хайку на заданную тему',
                    },
                    'syllables_per_line': {
                        'type': 'array',
                        'items': {'type': 'integer'},
                        'description': 'Количество слогов по строкам',
                    },
                    'total_words': {
                        'type': 'integer',
                        'description': 'Общее число слов в хайку',
                    },
                    'error': {
                        'type': 'string',
                        'description': 'Описание ошибки, если генерация не удалась',
                    },
                }
            },
        },
    ]
    tools_openai = [
        dict(
            type='function',
            function={
                k: v for k, v in t.items() if k in ('name', 'description', 'parameters')
            },
        )
        for t in tools_giga
    ]
    tools = tools_giga if config.insigma else tools_openai

    # payload

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            *message_history,
        ],
        'tools': tools,
        'temperature': config.freezing,
    }
    if config.insigma:
        payload['function_call'] = None  # 'auto'
        payload['functions'] = tools
        del payload['tools']

    # request

    response = post_chat_completions(payload, kwargs.get('verbose', False))

    # response

    if 'error' in response:
        logger.critical('select_tool_call // LLM Error: {}'.format(response['error']))
        return 2, None

    try:
        message = response['choices'][0]['message']

        function_call = message.get('function_call')
        if function_call:
            name = function_call.get('name')
            arguments = json.loads(function_call.get('arguments', '{}'))
            logger.debug('select_tool_call // Selection OK')
            return 0, (name, arguments)

        tool_calls = message.get('tool_calls')
        if tool_calls:
            tool_call = tool_calls[0]
            name = tool_call['function'].get('name')
            arguments = json.loads(tool_call['function'].get('arguments', '{}'))
            logger.debug('select_tool_call // Selection OK')
            return 0, (name, arguments)

        logger.warning('select_tool_call // LLM Selection Fail')
        return 1, None

    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as ex:
        logger.opt(exception=True).critical(
            f'select_tool_call // LLM Response Parse Error: {ex}'
        )
        return 3, None
