"""CLI agent with intent classification and tool routing."""

import json

from src.s02_simple_haiku.haiku.tool_haiku import (
    count_syllables_via_tool,
    generate_haiku,
)
from src.s02_simple_haiku.rag.tool_rag import answer_question
from src.utils_openai import post_chat_completions

EXIT_COMMANDS = {'exit', 'quit', 'q'}
HELP_COMMANDS = {'/help', 'help', '?'}


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


def classify_intent(user_input: str, **kwargs) -> bool:
    """
    Classify if user request is about Japanese poetry or haiku generation.
    """
    system_prompt = """Ты классификатор запросов.
Определи, относится ли запрос к японской поэзии (хайку/хокку) или генерации хайку.

Примеры запросов ДЛЯ НАШЕГО АГЕНТА (отвечай "да"):
- "напиши хайку о море"
- "сгенерируй хокку"
- "что такое хайку?"
- "объясни про хокку"
- "история жанра хайку"

Примеры запросов НЕ ДЛЯ НАШЕГО АГЕНТА (отвечай "нет"):
- "напиши стишок"
- "расскажи анекдот"
- "погода завтра"

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
        print(f'LLM Error: {response["error"]}')
        return False

    try:
        content = response['choices'][0]['message']['content'].strip().lower()
        return 'да' in content or 'yes' in content
    except (KeyError, IndexError) as e:
        print(f'LLM Response Error: {e}')
        return False


def select_tool_call(user_input: str) -> tuple[str, dict] | None:
    """
    Select tool via function calling for the given user input.
    """
    system_prompt = """Ты определяешь, какой инструмент вызвать.
Если пользователь спрашивает о японской поэзии, вызови rag_search.
Если пользователь просит сгенерировать хайку/хокку, вызови generate_haiku.
Вызывай ровно один инструмент."""

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'rag_search',
                'description': 'Поиск ответа на вопрос о японской поэзии',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'question': {
                            'type': 'string',
                            'description': 'Вопрос пользователя о хайку/хокку',
                        }
                    },
                    'required': ['question'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'generate_haiku',
                'description': 'Генерация хайку по теме',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'theme': {
                            'type': 'string',
                            'description': 'Тема хайку одним-двумя словами',
                        }
                    },
                    'required': ['theme'],
                },
            },
        },
    ]

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input},
        ],
        'tools': tools,
        'tool_choice': 'auto',
        'temperature': 0.0,
    }

    response = post_chat_completions(payload)
    if 'error' in response:
        print(f'LLM Error: {response["error"]}')
        return None

    try:
        message = response['choices'][0]['message']
    except (KeyError, IndexError) as e:
        print(f'LLM Response Error: {e}')
        return None

    tool_calls = message.get('tool_calls', [])
    if tool_calls:
        tool_call = tool_calls[0]
        function = tool_call.get('function', {})
        name = function.get('name')
        arguments = function.get('arguments', '{}')
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            args = {}
        return name, args

    function_call = message.get('function_call')
    if function_call:
        name = function_call.get('name')
        arguments = function_call.get('arguments', '{}')
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            args = {}
        return name, args

    print('LLM не выбрал инструмент.')
    return None


def detect_theme_change(user_input: str) -> str | None:
    """
    Detect theme change intent and extract new theme if provided.
    """
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
                print('Укажи новую тему после команды смены темы.')
            print('Сформулируй запрос заново.\n')
            return False

        if not theme:
            print('Не удалось определить тему для хайку.')
            return False

        if len(theme) > 60:
            print('Тема слишком длинная. Сократи до 1-2 слов.')
            return False

        return True

    print('Неизвестный инструмент.')
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

        print(f'\nВсего слогов: {stats.get("total_syllables", "?")}')
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
            answer = answer_question(question)
            print(f'\n{answer}\n')
            continue

        if tool_name == 'generate_haiku':
            theme = str(tool_args.get('theme', '')).strip()
            print(f'[Тема: {theme}]')
            haiku = generate_haiku(theme)

            if not haiku:
                print('Ошибка при генерации хайку.\n')
                continue

            stats = count_syllables_via_tool(haiku)
            display_haiku(haiku, stats)
            continue

        print('Не удалось выполнить инструмент.\n')


if __name__ == '__main__':
    main()
