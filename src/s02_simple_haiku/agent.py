"""CLI agent with intent classification and tool routing."""

import copy
import pprint

from ..config import config
from ..logger import logger
from .classify_intent import classify_intent
from .execute_gen_haiku import generate_haiku
from .execute_rag_search import answer_question
from .select_tool_call import select_tool_call
from .validate_tool_call import validate_tool_call

EXIT_COMMANDS = {'/exit', '/quit', '/q', 'exit', 'quit', 'q'}
HELP_COMMANDS = {'/help', 'help', '?'}
CLEAR_COMMANDS = {'/clear'}

VERBOSE = config.debug
CONTEXT_HIST_LIMIT = 10
MAX_CLARIFICATION_RETRIES = 3

RAG_TOP_K = 2


def get_help_message():
    """
    Print greeting and available capabilities.
    """
    return """
    === Мини-агент по японской поэзии ===
    Могу:
    - отвечать на вопросы о хайку/хокку
    - генерировать хайку по теме
    
    Для выхода введите: /exit, /quit или /q
    Очистить историю: /clear
    Команда справки: /help
    """


def add_to_history(history: list[dict], role: str, content: str):
    """
    Add message to history and trim to CONTEXT_HIST_LIMIT.
    """
    history.append({'role': role, 'content': content})

    # Обрезаем до последних CONTEXT_HIST_LIMIT сообщений
    if len(history) > CONTEXT_HIST_LIMIT:
        history[:] = history[-CONTEXT_HIST_LIMIT:]


def agent(message_history: list[dict]) -> dict:
    """
    Run classify/select/validate/execute pipeline for current history.
    """
    logger.debug('-' * 20 + ' agent // start ' + '-' * 20)
    message_history = copy.deepcopy(message_history)

    result: dict = {
        'last_state': None,
        'messages': [],
        'classify': {},
        'select': {},
        'validate': {},
        'execute': {},
        # 'message_history': message_history,
    }

    def append_message(message: str):
        result['messages'].append(message)

    # step1: classify

    append_message('[cls] Анализирую релевантность запроса..')
    result['last_state'] = 'classify'
    logger.debug(result)

    clf_result = classify_intent(message_history, verbose=VERBOSE)
    result['classify']['code'] = clf_result

    if clf_result == 103:
        clf_message = 'Ошибка при запросе LLM, завершаюсь..'
        result['classify']['message'] = clf_message
        append_message(f'[cls] {clf_message}')
        return result

    if clf_result == 102:
        clf_message = 'Ошибка при разборе ответа LLM, завершаюсь..'
        result['classify']['message'] = clf_message
        append_message(f'[cls] {clf_message}')
        return result

    if clf_result == 101:
        clf_message = 'Запрос не связан с функционалом агента.'
        result['classify']['message'] = clf_message
        append_message(f'[cls] {clf_message}')
        return result

    if clf_result == 100:
        clf_message = 'Запрос релевантен, думаю..'
        result['classify']['message'] = clf_message
        append_message(f'[cls] {clf_message}')
        add_to_history(message_history, 'assistant', clf_message)

    # step2: select

    append_message('[select] Выбираю подходящий инструмент..')
    result['last_state'] = 'select'
    logger.debug(result)

    select_result, select_tool = select_tool_call(message_history, verbose=VERBOSE)
    result['select']['code'] = select_result

    if select_tool:
        result['select']['tool_name'], result['select']['tool_args'] = select_tool

    if select_result == 203:
        select_message = 'Ошибка при запросе LLM, завершаюсь..'
        result['select']['message'] = select_message
        append_message(f'[select] {select_message}')
        return result

    if select_result == 202:
        select_message = 'Ошибка при разборе ответа LLM, завершаюсь..'
        result['select']['message'] = select_message
        append_message(f'[select] {select_message}')
        return result

    if select_result == 201:
        select_message = (
            'Не удалось определить инструмент. Просьба переформулировать запрос.'
        )
        result['select']['message'] = select_message
        append_message(f'[select] {select_message}')
        return result

    tool_name, tool_args = select_tool
    select_message = f'Выбран инструмент {tool_name} с параметрами {tool_args}'
    result['select']['message'] = select_message
    append_message(f'[select] {select_message}')

    # step3: validate

    append_message('[valid] Валидирую инструмент..')
    result['last_state'] = 'validate'
    logger.debug(result)

    valid_result, valid_info = validate_tool_call(tool_name, tool_args)

    result['validate']['code'] = 300 if valid_result else 301
    result['validate']['message'] = valid_info['message']

    if not valid_result:
        result['validate']['param'] = valid_info.get('param')
        result['validate']['reason'] = valid_info.get('reason')
        append_message(valid_info['message'])
        return result

    logger.info(f'[validate] {valid_info["message"]}')
    append_message(valid_info['message'])

    # step4: execute

    append_message('[exec] Выполняю инструмент..')
    result['last_state'] = 'execute'
    logger.debug(result)

    if tool_name == 'rag_search':
        question = tool_args['question'].strip()
        rag_result = answer_question(question, top_k=RAG_TOP_K)

        if 'error' in rag_result:
            if rag_result.get('error') == 'Health check failed':
                rag_message = (
                    'Небольшие трудности при запросе на RAG сервис.. '
                    'Инженеры уже работают над запуском сервиса..'
                )
            else:
                rag_message = (
                    'Произошла чудовищная ошибка при запросе на RAG сервис.. '
                    'Тысяча извинений! Попробуем снова?'
                )
            result['execute']['code'] = 401
            result['execute']['message'] = rag_message
            logger.info(f'[execute] {rag_message}')
            append_message(rag_message)
            return result

        rag_message = 'Ответ RAG: {}'.format(rag_result['answer'])
        result['execute']['code'] = 400
        result['execute']['message'] = rag_message
        logger.info(f'[execute] {rag_message}')
        append_message(rag_message)

        rag_titles_message = 'Заголовки топ-{} документов: {}'.format(
            RAG_TOP_K, ', '.join(rag_result['chunk_title_list'])
        )
        result['execute']['rag_titles_message'] = rag_message
        logger.info(f'[execute] {rag_titles_message}')

        rag_chunks = [
            f'[{i + 1}] {title}\n---{text}'
            for i, (title, text) in enumerate(
                zip(rag_result['chunk_title_list'], rag_result['chunk_texts'])
            )
        ]
        rag_chunks_message = '\n\n'.join(rag_chunks)
        result['execute']['rag_chunks_message'] = rag_chunks_message
        logger.debug(f'[execute] rag_chunks_message:\n{rag_chunks_message}')
        return result

    if tool_name == 'generate_haiku':
        theme = str(tool_args['theme']).strip()
        result_haiku = generate_haiku(theme)

        if 'error' in result_haiku:
            if result_haiku.get('error') == 'Health check failed':
                haiku_message = (
                    'Небольшие трудности при генерации хайку.. '
                    'Инженеры уже работают над запуском сервиса..'
                )
            else:
                haiku_message = (
                    'Произошла чудовищная ошибка при генерации хайку.. '
                    'Тысяча извинений! Попробуем снова?'
                )
            result['execute']['code'] = 401
            result['execute']['message'] = haiku_message
            logger.info(f'[execute] {haiku_message}')
            append_message(haiku_message)
            return result

        haiku_text = result_haiku['haiku_text'].strip()
        syllables_per_line = result_haiku['syllables_per_line']
        total_words = result_haiku['total_words']

        haiku_message = 'Хайку: {}'.format(
            ' | '.join(
                [line.strip() for line in haiku_text.splitlines() if line.strip()]
            )
        )
        result['execute']['code'] = 400
        result['execute']['message'] = haiku_message
        logger.info(f'[execute] {haiku_message}')
        append_message(haiku_message)

        syllables_msg = (
            '-'.join(str(value) for value in syllables_per_line)
            if syllables_per_line
            else '?'
        )
        result['execute']['syllables_msg'] = syllables_msg
        result['execute']['total_words'] = total_words
        logger.info(f'[execute] #слогов построчно: {syllables_msg}')
        logger.info(f'[execute] #слов итого: {total_words}')
        return result

    fallback_message = 'Что-то пошло не так.. Начнем с чистого листа!'
    result['execute']['code'] = 402
    result['execute']['message'] = fallback_message
    logger.info(f'[execute] {fallback_message}')
    append_message(fallback_message)
    return result


def main():
    """
    Main interactive loop for the haiku agent.
    """
    message_history: list[dict] = []
    iteration = 0

    while True:
        if iteration == 0:
            logger.debug({'state': 'AgentStart'})
            logger.info(get_help_message())
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
            logger.info(get_help_message())
            continue

        if lowered in CLEAR_COMMANDS:
            message_history.clear()
            logger.info('[main] История сообщений очищена.')
            logger.debug({'state': 'AgentClear'})
            continue

        add_to_history(message_history, 'user', user_input)

        agent_result = agent(message_history)

        classify_info = agent_result.get('classify') or {}
        select_info = agent_result.get('select') or {}
        validate_info = agent_result.get('validate') or {}
        execute_info = agent_result.get('execute') or {}
        think_messages = agent_result.get('messages') or []

        if (
            classify_info.get('code') != 100
            or select_info.get('code') != 200
            or validate_info.get('code') != 300
            or execute_info.get('code') != 400
        ):
            logger.info(
                '[main] Неожиданный результат (завершаюсь):\n{}'.format(
                    pprint.pformat(agent_result, width=120, sort_dicts=False)
                )
            )
            logger.debug({'state': 'AgentEnd'})
            break

        add_to_history(message_history, 'assistant', agent_result['execute']['message'])

        # for msg in messages:
        #     add_to_history(message_history, 'assistant', msg)
        #     logger.info(msg)


if __name__ == '__main__':
    main()
