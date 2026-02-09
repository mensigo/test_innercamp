"""CLI agent with intent classification and tool routing."""

import json

from src import config, post_chat_completions

from .haiku import generate_haiku
from .rag import answer_question

EXIT_COMMANDS = {'exit', 'quit', 'q'}
HELP_COMMANDS = {'/help', 'help', '?'}

verbose = False


def print_help():
    """
    Print greeting and available capabilities.
    """
    print('=== Мини-агент по японской поэзии ===')
    print('Могу:')
    print('- отвечать на вопросы о хайку/хокку (RAG)')
    print('- генерировать хайку по теме')
    print('Для выхода введите: exit, quit или q')
    print('Команда помощи: /help\n')


def print_reminder():
    """
    Remind available capabilities.
    """
    print('Не совсем понимаю запрос. Мои возможности:')
    print('- отвечать на вопросы о хайку/хокку (RAG)')
    print('- генерировать хайку по теме')


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
        'temperature': kwargs.get('temperature', 0.001),
        'max_tokens': 1,
    }

    response = post_chat_completions(payload)

    if 'error' in response:
        print('[classify_intent] LLM Error: {}'.format(response['error']))
        return False

    try:
        content = response['choices'][0]['message']['content'].strip().lower()
        return 'да' in content or 'yes' in content

    except (KeyError, IndexError) as e:
        print(f'[classify_intent] LLM Response Error: {e}')
        return False


def select_tool_call(user_input: str) -> tuple[str, dict] | None:
    """
    Select tool via function calling for the given user input.
    Return name,args if succeeded, None otherwise.
    """
    print('[select_tool_call] start')

    # system
    system_prompt = """Ты определяешь, какой инструмент вызвать.
Если пользователь спрашивает о японской поэзии, вызови rag_search.
Если пользователь просит сгенерировать хайку/хокку, вызови generate_haiku.
Вызывай ровно один инструмент."""

    if not config.insigma:
        system_prompt += """

Примеры взаимодействия (generate_haiku):
Пользователь: Создай хайку о зимнем утре
Ты: generate_haiku({"theme": "Зимнее утро"})

Примеры взаимодействия (rag_search):
Пользователь: Когда был написан манъесю
Ты: rag_search({"question": "Дата написания Манъёсю"})
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
                # 'required': ['question'],
            },
            'few_shot_examples': [
                {
                    'request': 'Есть ли в старояпонском священные цифры?',
                    'params': {
                        'question': 'Старояпонский цифры в старояпонском языке?'
                    },
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
                # 'required': ['theme'],
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
        {k: v for k, v in t.items() if k in ('name', 'description', 'parameters')}
        for t in tools_giga
    ]
    tools = tools_giga if config.insigma else tools_openai
    allowed_tool_names = {tool['name'] for tool in tools}

    # payload
    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input},
        ],
        'funcitons': tools,
        'temperature': config.freezing,
    }
    if config.insigma:
        payload['function_call'] = 'auto'

    # request
    response = post_chat_completions(payload, verbose)
    if 'error' in response:
        print(f'[select_tool_call] LLM Error: {response["error"]}')
        return None

    # handle response
    message = response['choices'][0]['message']
    function_call = message.get('function_call')

    if function_call:
        name = function_call.get('name')
        arguments = function_call.get('arguments', '{}')
        if name not in allowed_tool_names:
            print(
                '[select_tool_call] LLM вернул неизвестный инструмент: {}'.format(name)
            )
            return None
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return name, arguments

    print('[select_tool_call] LLM не выбрал инструмент')
    return None


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


def validate_tool_call(tool_name: str, tool_args: dict, user_input: str) -> bool:
    """
    Validate tool arguments before calling tool.
    """
    print('[validate_tool_call] start')

    if tool_name == 'rag_search':
        question = str(tool_args.get('question', '')).strip()
        if not question:
            print('Не удалось определить вопрос для RAG.')
            return False
        return True

    if tool_name == 'generate_haiku':
        theme = str(tool_args.get('theme', '')).strip()
        theme_change = detect_theme_change(user_input)
        if theme_change is not None:
            if theme_change:
                print(f'Тема изменена на: {theme_change}')
            else:
                print('Укажите новую тему после команды смены темы.')
            print('Сформулируй запрос заново.\n')
            return False

        if not theme:
            print('Не удалось определить тему для хайку.')
            return False

        if len(theme) > 20:
            print('Тема слишком длинная. Сократи до 1-2 слов.')
            return False

        return True

    print('[validate_tool_call] Неизвестный инструмент.')
    return False


def display_haiku(haiku: str, stats: dict | None):
    """
    Display haiku with syllable counts if available.
    """
    if not haiku:
        print('Не удалось сгенерировать хайку.')
        return

    lines = haiku.strip().split('\n')
    print('\n--- Хайку ---')

    if stats and 'syllables_per_line' in stats:
        syllable_counts = stats['syllables_per_line']
        for i, line in enumerate(lines):
            if i < len(syllable_counts):
                print(f'{line.strip()} ({syllable_counts[i]})')
            else:
                print(line.strip())

        total_syllables = sum(syllable_counts) if syllable_counts else 0
        print(f'\nВсего слогов: {total_syllables}')
        print(f'Всего слов: {stats.get("total_words", "?")}')
    else:
        for line in lines:
            print(line.strip())

    print('-------------\n')


def main():
    """
    Main interactive loop for the haiku agent.
    """
    print_help()

    while True:
        user_input = input('Введите запрос: ').strip()

        # TODO: check length (deny if too long)

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in EXIT_COMMANDS:
            print('До свидания!')
            break

        if lowered in HELP_COMMANDS:
            print_help()
            continue

        if not classify_intent(user_input):
            print('CLF: не наш агент\n')
            print_reminder()
            continue

        tool_call = select_tool_call(user_input)
        if not tool_call:
            print('Не удалось определить инструмент.\n')
            continue

        tool_name, tool_args = tool_call
        if not validate_tool_call(tool_name, tool_args, user_input):
            continue

        if tool_name == 'rag_search':
            question = str(tool_args.get('question', '')).strip()
            response = answer_question(question)
            answer = response.get('answer', '')
            print(f'\n{answer}\n')
            continue

        if tool_name == 'generate_haiku':
            theme = str(tool_args.get('theme', '')).strip()
            print(f'[Тема: {theme}]')
            result = generate_haiku(theme)

            if 'error' in result and result['error']:
                print(f'Ошибка при генерации хайку: {result["error"]}\n')
                continue

            haiku_text = result.get('haiku_text', '')
            if not haiku_text:
                print('Ошибка при генерации хайку.\n')
                continue

            display_haiku(haiku_text, result)
            continue

        print('Не удалось выполнить инструмент.\n')


if __name__ == '__main__':
    main()
