"""CLI agent with intent classification and tool routing."""

import copy
import pprint
from pathlib import Path
import readline
from typing import Generator

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

RAG_TOP_K = 2
# Store CLI history next to agent.log in repo root
HISTORY_PATH = Path('agent_history.txt')


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


def load_cli_history():
    """
    Load CLI history from file for readline support.
    """
    try:
        if HISTORY_PATH.exists():
            # Let readline parse its own format (avoids \040 escapes)
            readline.read_history_file(HISTORY_PATH)

            # Deduplicate consecutive duplicates after load
            items = [
                readline.get_history_item(i)
                for i in range(1, readline.get_current_history_length() + 1)
            ]
            deduped: list[str] = []
            for item in items:
                if item and (not deduped or deduped[-1] != item):
                    deduped.append(item)

            readline.clear_history()
            for item in deduped:
                readline.add_history(item)

            persist_cli_history()
    except Exception as ex:
        logger.warning(f'agent // Failed to load history: {ex}')


def persist_cli_history():
    """
    Persist current readline history to file.
    """
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(HISTORY_PATH)
    except Exception as ex:
        logger.warning(f'agent // Failed to save history: {ex}')


def append_cli_history(entry: str):
    """
    Append entry to readline history and persist.
    """
    if not entry:
        return
    try:
        last_idx = readline.get_current_history_length()
        last_entry = readline.get_history_item(last_idx) if last_idx else None
        if last_entry != entry:
            readline.add_history(entry)
            persist_cli_history()
    except Exception as ex:
        logger.warning(f'agent // Failed to append history: {ex}')


def agent_yield(message_history: list[dict]) -> Generator[str, None, dict]:
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
    }

    def add_message(message: str) -> str:
        result['messages'].append(message)
        return message

    # step1: classify

    yield add_message('[cls] Анализирую релевантность запроса..')
    result['last_state'] = 'classify'
    logger.debug(result)

    clf_result = classify_intent(message_history, verbose=VERBOSE)
    result['classify']['code'] = clf_result

    if clf_result == 103:
        clf_message = 'Ошибка при запросе LLM, завершаюсь..'
        result['classify']['message'] = clf_message
        yield add_message(f'[cls] {clf_message}')
        return result

    if clf_result == 102:
        clf_message = 'Ошибка при разборе ответа LLM, завершаюсь..'
        result['classify']['message'] = clf_message
        yield add_message(f'[cls] {clf_message}')
        return result

    if clf_result == 101:
        clf_message = 'Запрос не связан с функционалом агента.'
        result['classify']['message'] = clf_message
        # add_to_history(message_history, 'assistant', clf_message)
        yield add_message(f'[cls] {clf_message}')
        return result

    if clf_result == 100:
        clf_message = 'Запрос релевантен, думаю..'
        result['classify']['message'] = clf_message
        add_to_history(message_history, 'assistant', clf_message)
        yield add_message(f'[cls] {clf_message}')

    # step2: select

    yield add_message('[select] Выбираю подходящий инструмент..')
    result['last_state'] = 'select'
    logger.debug(result)

    select_result, select_tool = select_tool_call(message_history, verbose=VERBOSE)
    result['select']['code'] = select_result

    if select_tool:
        result['select']['tool_name'], result['select']['tool_args'] = select_tool

    if select_result == 203:
        select_message = 'Ошибка при запросе LLM, завершаюсь..'
        result['select']['message'] = select_message
        yield add_message(f'[select] {select_message}')
        return result

    if select_result == 202:
        select_message = 'Ошибка при разборе ответа LLM, завершаюсь..'
        result['select']['message'] = select_message
        yield add_message(f'[select] {select_message}')
        return result

    if select_result == 201:
        select_message = (
            'Не удалось определить инструмент. Просьба переформулировать запрос.'
        )
        result['select']['message'] = select_message
        # add_to_history(message_history, 'assistant', select_message)
        yield add_message(f'[select] {select_message}')
        return result

    tool_name, tool_args = select_tool
    select_message = f'Выбран инструмент {tool_name} с параметрами {tool_args}'
    result['select']['message'] = select_message
    yield add_message(f'[select] {select_message}')

    # step3: validate

    yield add_message('[valid] Валидирую инструмент..')
    result['last_state'] = 'validate'
    logger.debug(result)

    valid_result, valid_info = validate_tool_call(tool_name, tool_args)

    result['validate']['code'] = 300 if valid_result else 301
    result['validate']['message'] = valid_info['message']

    if not valid_result:
        result['validate']['param'] = valid_info.get('param')
        result['validate']['reason'] = valid_info.get('reason')
        yield add_message(f'[valid] {valid_info["message"]}')
        return result

    yield add_message('[valid] ' + valid_info['message'])

    # step4: execute

    yield add_message('[exec] Выполняю инструмент..')
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
            yield add_message(f'[exec] {rag_message}')
            return result

        rag_message = 'Ответ RAG: {}'.format(rag_result['answer'])
        result['execute']['code'] = 400
        result['execute']['message'] = rag_message
        yield add_message(f'[exec] {rag_message}')

        chunk_titles = rag_result.get('chunk_title_list') or []
        chunk_sources = rag_result.get('chunk_source_list') or []

        rag_titles_message = 'Заголовки топ-{} документов: {}'.format(
            RAG_TOP_K, ', '.join(chunk_titles)
        )
        result['execute']['rag_titles_message'] = rag_titles_message
        logger.debug(f'[exec] {rag_titles_message}')
        yield add_message('[exec] ' + rag_titles_message)

        if chunk_sources:
            rag_sources_message = 'Файлы топ-{} документов: {}'.format(
                RAG_TOP_K, ', '.join(chunk_sources)
            )
            result['execute']['rag_sources_message'] = rag_sources_message
            logger.debug(f'[exec] {rag_sources_message}')
            yield add_message('[exec] ' + rag_sources_message)

        rag_chunks = [
            f'[{i + 1}] Title: {title} | File: {chunk_sources[i] if i < len(chunk_sources) else "?"}\nText: {text}'
            for i, (title, text) in enumerate(
                zip(chunk_titles, rag_result['chunk_texts'])
            )
        ]
        rag_chunks_message = '-----\n' + '\n-----\n'.join(rag_chunks)
        result['execute']['rag_chunks_message'] = rag_chunks_message
        logger.debug('[exec] rag_chunks_message:\n' + rag_chunks_message)
        yield add_message('[exec] ' + rag_chunks_message)
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
            yield add_message(f'[exec] {haiku_message}')
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
        logger.debug(f'[exec] {haiku_message}')
        yield add_message('[exec] ' + haiku_message)

        syllables_msg = (
            '-'.join(str(value) for value in syllables_per_line)
            if syllables_per_line
            else '?'
        )
        result['execute']['syllables_msg'] = syllables_msg
        result['execute']['total_words'] = total_words
        logger.debug(f'[exec] #слогов построчно: {syllables_msg}')
        logger.debug(f'[exec] #слов итого: {total_words}')
        return result

    fallback_message = 'Что-то пошло не так.. Начнем с чистого листа!'
    result['execute']['code'] = 402
    result['execute']['message'] = fallback_message
    logger.debug(f'[exec] {fallback_message}')
    yield add_message('[exec] ' + fallback_message)
    return result


def agent(message_history: list[dict]) -> dict:
    """
    Run agent pipeline and return final result.
    """
    generator = agent_yield(message_history)
    try:
        while True:
            msg = next(generator)
            logger.info(msg)
    except StopIteration as ex:
        return ex.value or {}


def main():
    """
    Main interactive loop for the haiku agent.
    """
    message_history: list[dict] = []
    iteration = 0

    load_cli_history()
    readline.set_history_length(50)

    try:
        while True:
            if iteration == 0:
                logger.debug('AgentStart')
                print(get_help_message())
            else:
                logger.debug('AgentRestart')

            user_input = input('user: ').strip()
            iteration += 1

            if not user_input:
                continue

            append_cli_history(user_input)

            lowered = user_input.lower()
            if lowered in EXIT_COMMANDS:
                logger.debug('AgentEnd')
                print('assistant: До свидания!')
                break

            if lowered in HELP_COMMANDS:
                logger.debug('AgentHelp')
                print(get_help_message())
                continue

            if lowered in CLEAR_COMMANDS:
                message_history.clear()
                readline.clear_history()
                try:
                    HISTORY_PATH.unlink(missing_ok=True)
                except Exception as ex:
                    logger.warning(f'agent // Failed to delete history file: {ex}')
                persist_cli_history()
                logger.debug('AgentClear')
                print('assistant: История сообщений очищена.')
                continue

            add_to_history(message_history, 'user', user_input)

            agent_result = agent(message_history)

            classify_info = agent_result.get('classify') or {}
            select_info = agent_result.get('select') or {}
            validate_info = agent_result.get('validate') or {}
            execute_info = agent_result.get('execute') or {}
            think_messages = agent_result.get('messages') or []

            # if irrelevant query, print message and reask user
            if (
                agent_result.get('last_state') == 'classify'
                and classify_info.get('code') == 101
            ):
                print(f'assistant: {agent_result["classify"]["message"]}')
                add_to_history(
                    message_history, 'assistant', agent_result['classify']['message']
                )
                continue

            # if tool not determined, print message and reask user
            if (
                agent_result.get('last_state') == 'select'
                and select_info.get('code') == 201
            ):
                print(f'assistant: {agent_result["select"]["message"]}')
                add_to_history(
                    message_history, 'assistant', agent_result['select']['message']
                )
                continue

            # if validation failed, print message and reask user
            if (
                agent_result.get('last_state') == 'validate'
                and validate_info.get('code') == 301
            ):
                print(f'assistant: {agent_result["validate"]["message"]}')
                add_to_history(
                    message_history, 'assistant', agent_result['validate']['message']
                )
                continue

            # if unexpected result, print message and exit
            if (
                classify_info.get('code') != 100
                or select_info.get('code') != 200
                or validate_info.get('code') != 300
                or execute_info.get('code') != 400
            ):
                print(
                    'assistant: Неожиданный результат (завершаюсь):\n{}'.format(
                        pprint.pformat(agent_result, width=120, sort_dicts=False)
                    )
                )
                logger.debug('AgentEnd')
                break

            print(f'assistant: {agent_result["execute"]["message"]}')
            add_to_history(
                message_history, 'assistant', agent_result['execute']['message']
            )
    finally:
        persist_cli_history()


if __name__ == '__main__':
    main()
