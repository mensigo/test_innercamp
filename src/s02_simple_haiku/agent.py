"""CLI agent with intent classification and tool routing."""

from .classify_intent import classify_intent
from .haiku import generate_haiku
from .rag import answer_question
from .select_tool_call import select_tool_call

EXIT_COMMANDS = {'exit', 'quit', 'q'}
HELP_COMMANDS = {'/help', 'help', '?'}

VERBOSE = True


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

        if not classify_intent(user_input, verbose=VERBOSE):
            print('CLF: не наш агент\n')
            print_reminder()
            continue

        tool_call = select_tool_call(user_input, verbose=VERBOSE)
        if not tool_call:
            print('SEL: Не удалось определить инструмент.\n')
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
