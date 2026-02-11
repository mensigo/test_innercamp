"""CLI agent with intent classification and tool routing."""

from .clarification import (
    extract_param_from_clarification,
    generate_clarification_prompt,
)
from .classify_intent import classify_intent
from .haiku import generate_haiku
from .rag import answer_question
from .select_tool_call import select_tool_call
from .validate_tool_call import validate_tool_call

EXIT_COMMANDS = {'exit', 'quit', 'q'}
HELP_COMMANDS = {'/help', 'help', '?'}

VERBOSE = True
CONTEXT_HIST_LIMIT = 10
MAX_CLARIFICATION_RETRIES = 3


def print_help():
    """
    Print greeting and available capabilities.
    """
    print('[print_help | info] === Мини-агент по японской поэзии ===')
    print('[print_help | info] Могу:')
    print('[print_help | info] - отвечать на вопросы о хайку/хокку (RAG)')
    print('[print_help | info] - генерировать хайку по теме')
    print('[print_help | info] Для выхода введите: exit, quit или q')
    print('[print_help | info] Команда помощи: /help\n')


def print_reminder():
    """
    Remind available capabilities.
    """
    print('[print_reminder | info] Не совсем понимаю запрос. Мои возможности:')
    print('[print_reminder | info] - отвечать на вопросы о хайку/хокку (RAG)')
    print('[print_reminder | info] - генерировать хайку по теме')


def add_to_history(history: list[dict], role: str, content: str):
    """
    Add message to history and trim to CONTEXT_HIST_LIMIT.
    """
    history.append({'role': role, 'content': content})

    # Обрезаем до последних CONTEXT_HIST_LIMIT сообщений
    if len(history) > CONTEXT_HIST_LIMIT:
        history[:] = history[-CONTEXT_HIST_LIMIT:]


def display_haiku(haiku: str, stats: dict | None):
    """
    Display haiku with syllable counts if available.
    """
    if not haiku:
        print('[display_haiku | error] Не удалось сгенерировать хайку.')
        return

    lines = haiku.strip().split('\n')
    print('[display_haiku | result] \n--- Хайку ---')

    if stats and 'syllables_per_line' in stats:
        syllable_counts = stats['syllables_per_line']
        for i, line in enumerate(lines):
            if i < len(syllable_counts):
                print(f'[display_haiku | result] {line.strip()} ({syllable_counts[i]})')
            else:
                print(f'[display_haiku | result] {line.strip()}')

        total_syllables = sum(syllable_counts) if syllable_counts else 0
        print(f'[display_haiku | result] \nВсего слогов: {total_syllables}')
        print(f'[display_haiku | result] Всего слов: {stats.get("total_words", "?")}')
    else:
        for line in lines:
            print(f'[display_haiku | result] {line.strip()}')

    print('[display_haiku | result] -------------\n')


def main():
    """
    Main interactive loop for the haiku agent.
    """
    print_help()

    message_history: list[dict] = []
    pending_clarification: dict | None = None

    while True:
        user_input = input('Введите запрос: ').strip()

        # TODO: check length (deny if too long)

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in EXIT_COMMANDS:
            print('[main | exit] До свидания!')
            break

        if lowered in HELP_COMMANDS:
            print_help()
            continue

        # Добавляем пользовательский ввод в историю
        add_to_history(message_history, 'user', user_input)

        # Если ожидаем уточнение параметра
        if pending_clarification:
            print('[main | clarification] Обработка уточнения...')
            extracted = extract_param_from_clarification(
                user_input,
                pending_clarification['missing_param'],
                pending_clarification['tool_name'],
            )

            if extracted:
                print(f'[main | clarification] Параметр извлечен: {extracted}')
                # Обновляем аргументы инструмента
                tool_name = pending_clarification['tool_name']
                tool_args = pending_clarification['tool_args']
                tool_args[pending_clarification['missing_param']] = extracted

                # Выполняем инструмент
                if tool_name == 'rag_search':
                    print('[main | execute_tool] Выполнение rag_search')
                    question = str(tool_args.get('question', '')).strip()
                    response = answer_question(question)
                    answer = response.get('answer', '')
                    print(f'[main | execute_tool] \n{answer}\n')
                    add_to_history(message_history, 'assistant', answer)

                elif tool_name == 'generate_haiku':
                    print('[main | execute_tool] Выполнение generate_haiku')
                    theme = str(tool_args.get('theme', '')).strip()
                    print(f'[main | execute_tool] Тема: {theme}')
                    result = generate_haiku(theme)

                    if 'error' in result and result['error']:
                        error_msg = f'Ошибка при генерации хайку: {result["error"]}'
                        print(f'[main | execute_tool] {error_msg}\n')
                        add_to_history(message_history, 'assistant', error_msg)
                    else:
                        haiku_text = result.get('haiku_text', '')
                        if not haiku_text:
                            error_msg = 'Ошибка при генерации хайку.'
                            print(f'[main | execute_tool] {error_msg}\n')
                            add_to_history(message_history, 'assistant', error_msg)
                        else:
                            display_haiku(haiku_text, result)
                            add_to_history(message_history, 'assistant', haiku_text)

                pending_clarification = None
            else:
                # Параметр не извлечен, увеличиваем счетчик попыток
                pending_clarification['retry_count'] += 1
                print(
                    f'[main | clarification] Попытка {pending_clarification["retry_count"]}/{MAX_CLARIFICATION_RETRIES}'
                )

                if pending_clarification['retry_count'] >= MAX_CLARIFICATION_RETRIES:
                    response = 'Не удалось получить параметр. Попробуйте заново.'
                    print(f'[main | clarification] {response}\n')
                    add_to_history(message_history, 'assistant', response)
                    pending_clarification = None
                else:
                    # Повторяем уточняющий вопрос
                    prompt = generate_clarification_prompt(
                        pending_clarification['tool_name'],
                        pending_clarification['missing_param'],
                    )
                    print(f'[main | clarification] {prompt}')
                    add_to_history(message_history, 'assistant', prompt)

            continue

        # Обычный flow: classify -> select -> validate -> execute

        if not classify_intent(user_input, verbose=VERBOSE):
            response = 'CLF: не наш агент'
            print(f'[main | classify_intent] {response}\n')
            add_to_history(message_history, 'assistant', response)
            print_reminder()
            continue

        tool_call = select_tool_call(user_input, verbose=VERBOSE)
        if not tool_call:
            response = 'SEL: Не удалось определить инструмент. Переформулируйте вопрос.'
            print(f'[main | select_tool] {response}\n')
            add_to_history(message_history, 'assistant', response)
            continue

        tool_name, tool_args = tool_call
        is_valid, missing_param = validate_tool_call(tool_name, tool_args, user_input)

        if not is_valid:
            if missing_param:
                # Запускаем цикл уточнения
                prompt = generate_clarification_prompt(tool_name, missing_param)
                print(f'[main | clarification] {prompt}')
                add_to_history(message_history, 'assistant', prompt)
                pending_clarification = {
                    'tool_name': tool_name,
                    'tool_args': tool_args,
                    'missing_param': missing_param,
                    'retry_count': 0,
                    'original_user_input': user_input,
                }
            else:
                # Другая ошибка валидации (уже выведена в validate_tool_call)
                response = 'Некорректный запрос. Попробуйте снова.'
                print(f'[main | validate_tool] {response}\n')
                add_to_history(message_history, 'assistant', response)
            continue

        # Выполняем инструмент
        if tool_name == 'rag_search':
            print('[main | execute_tool] Выполнение rag_search')
            question = str(tool_args.get('question', '')).strip()
            response = answer_question(question)
            answer = response.get('answer', '')
            print(f'[main | execute_tool] \n{answer}\n')
            add_to_history(message_history, 'assistant', answer)
            continue

        if tool_name == 'generate_haiku':
            print('[main | execute_tool] Выполнение generate_haiku')
            theme = str(tool_args.get('theme', '')).strip()
            print(f'[main | execute_tool] Тема: {theme}')
            result = generate_haiku(theme)

            if 'error' in result and result['error']:
                error_msg = f'Ошибка при генерации хайку: {result["error"]}'
                print(f'[main | execute_tool] {error_msg}\n')
                add_to_history(message_history, 'assistant', error_msg)
                continue

            haiku_text = result.get('haiku_text', '')
            if not haiku_text:
                error_msg = 'Ошибка при генерации хайку.'
                print(f'[main | execute_tool] {error_msg}\n')
                add_to_history(message_history, 'assistant', error_msg)
                continue

            display_haiku(haiku_text, result)
            add_to_history(message_history, 'assistant', haiku_text)
            continue

        response = 'Не удалось выполнить инструмент.'
        print(f'[main | execute_tool] {response}\n')
        add_to_history(message_history, 'assistant', response)


if __name__ == '__main__':
    main()
