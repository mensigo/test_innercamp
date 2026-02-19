"""CLI agent with intent classification and tool routing."""

from ..config import config
from ..logger import logger
from .classify_intent import classify_intent
from .haiku import generate_haiku
from .rag import answer_question
from .select_tool_call import select_tool_call
from .validate_tool_call import validate_tool_call

EXIT_COMMANDS = {'/exit', '/quit', '/q', 'exit', 'quit', 'q'}
HELP_COMMANDS = {'/help', 'help', '?'}

VERBOSE = config.debug
CONTEXT_HIST_LIMIT = 10
MAX_CLARIFICATION_RETRIES = 3

RAG_TOP_K = 2


def print_help():
    """
    Print greeting and available capabilities.
    """
    logger.info("""
    === Мини-агент по японской поэзии ===
    Могу:
    - отвечать на вопросы о хайку/хокку
    - генерировать хайку по теме
    
    Для выхода введите: /exit, /quit или /q
    Команда справки: /help
    """)


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
    message_history: list[dict] = []
    iteration = 0

    while True:
        if iteration == 0:
            logger.debug({'state': 'AgentStart'})
            print_help()
        else:
            logger.debug({'state': 'AgentRestart'})

        user_input = input('Введите запрос: ').strip()
        iteration += 1

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in EXIT_COMMANDS:
            logger.info('[main] До свидания!')
            logger.debug({'state': 'AgentEnd'})
            break

        if lowered in HELP_COMMANDS:
            logger.debug({'state': 'AgentHelp'})
            print_help()
            continue

        add_to_history(message_history, 'user', user_input)

        # Обычный flow: classify -> select -> validate -> execute

        # classify

        logger.info('[cls] Анализирую релевантность запроса..')
        logger.debug({'state': 'AgentClassify'})

        clf_result = classify_intent(message_history, verbose=VERBOSE)

        if clf_result == 3:
            clf_message = 'Ошибка при разборе ответа LLM, завершаюсь..'
            logger.info(f'[cls] {clf_message}\n')
            break

        if clf_result == 2:
            clf_message = 'Ошибка при запросе LLM, завершаюсь..'
            logger.info(f'[cls] {clf_message}\n')
            break

        if clf_result == 1:
            clf_message = 'Запрос не связан с функционалом агента.'
            logger.info(f'[cls] {clf_message}\n')
            print_help()
            add_to_history(message_history, 'assistant', clf_message)
            continue

        if clf_result == 0:
            clf_message = 'Запрос релевантен, думаю..'
            logger.info(f'[cls] {clf_message}\n')
            add_to_history(message_history, 'assistant', clf_message)

        # select

        logger.info('[select] Выбираю подходящий инструмент..')
        logger.debug({'state': 'AgentSelect'})

        select_result, select_tool = select_tool_call(message_history, verbose=VERBOSE)

        if select_result == 3:
            select_message = 'Ошибка при разборе ответа LLM, завершаюсь..'
            logger.info(f'[select] {select_message}\n')
            break

        if select_result == 2:
            select_message = 'Ошибка при запросе LLM, завершаюсь..'
            logger.info(f'[select] {select_message}\n')
            break

        if select_result == 1:
            select_message = (
                'Не удалось определить инструмент. Просьба переформулировать запрос.'
            )
            logger.info(f'[select] {select_message}\n')
            add_to_history(message_history, 'assistant', select_message)
            continue

        if select_result == 0:
            tool_name, tool_args = select_tool
            select_message = f'Выбран инструмент {tool_name} с параметрами {tool_args}'
            logger.info(f'[select] {select_message}\n')
            add_to_history(message_history, 'assistant', select_message)

        # validate

        logger.info('[valid] Валидирую инструмент..')
        logger.debug({'state': 'AgentValidate'})

        valid_result, valid_info = validate_tool_call(tool_name, tool_args)

        if valid_result:
            valid_message = valid_info['message']
            logger.info(f'[valid] {valid_message}\n')
            add_to_history(message_history, 'assistant', valid_message)

        else:
            fail_message = valid_info['message']
            logger.info(f'[valid] {fail_message}')
            add_to_history(message_history, 'assistant', fail_message)
            continue

        # execute

        logger.info('[exec] Выполняю инструмент..')
        logger.debug({'state': 'AgentExecute'})

        if tool_name == 'rag_search':
            question = tool_args['question'].strip()
            rag_result = answer_question(question, top_k=RAG_TOP_K)

            if 'error' in rag_result:
                rag_message = 'Произошла чудовищная ошибка при запросе на RAG сервис.. Тысяча извинений!'
                logger.info(f'[exec] {rag_message}')
                add_to_history(message_history, 'assistant', rag_message)
                continue

            rag_message = 'Ответ RAG: {}'.format(response['answer'])
            logger.info(f'[exec] {rag_message}')
            add_to_history(message_history, 'assistant', rag_message)

            rag_titles_message = 'Заголовки топ-{} документов: {}'.format(
                RAG_TOP_K, ', '.join(response['chunk_title_list'])
            )
            logger.info(f'[exec] {rag_titles_message}')

            rag_chunks = [
                f'[{i + 1}] {title}\n---{text}'
                for i, (title, text) in enumerate(
                    zip(response['chunk_title_list'], response['chunk_texts'])
                )
            ]
            rag_chunks_message = '\n\n'.join(rag_chunks)
            logger.debug(f'[exec] rag_chunks_message:\n{rag_chunks_message}')

        # WIP

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
