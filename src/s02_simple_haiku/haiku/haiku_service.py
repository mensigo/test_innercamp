"""Flask service for haiku generation."""

from flask import Flask, jsonify, request

from src import config, post_chat_completions

from .count_stats import count_syllables_and_words
from .logger import logger

app = Flask(__name__)


def generate_haiku(theme: str, **kwargs) -> str:
    """
    Generate Russian haiku on specified topic using LLM.
    """
    system_prompt = """Ты поэт, который пишет хайку на русском языке.

СТРОГИЕ ТРЕБОВАНИЯ:
1. Формат 5-7-5 слогов (первая строка - 5 слогов, вторая - 7 слогов, третья - 5 слогов)
2. Структура:
   - Строка 1: Короткий образ (5 слогов)
   - Строка 2: Развитие образа, сопоставление (7 слогов)
   - Строка 3: Второй образ, неожиданная связь или вывод (5 слогов)
3. Используй тему, указанную пользователем
4. Пиши в стиле традиционного японского хайку
5. Выводи ТОЛЬКО текст хайку, по одной строке, БЕЗ нумерации, БЕЗ дополнительных комментариев

Пример хайку:
Море в тишине
Волны шепчут о вечном
Закат золотой"""

    user_prompt = f'Напиши хайку на тему: {theme}'

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': kwargs.get('temperature', 0.5),
    }

    response = post_chat_completions(payload, kwargs.get('verbose', False))

    if 'error' in response:
        logger.error('haiku_service // LLM Error: {}'.format(response['error']))
        return ''

    haiku = response['choices'][0]['message']['content'].strip()
    return haiku


@app.route('/generate_haiku', methods=['POST'])
def generate_haiku_endpoint():
    """
    Generate haiku on given theme and return with syllable stats.
    """
    data = request.get_json()

    if not data or 'theme' not in data:
        logger.error('haiku_service // generation error: missing theme')
        return jsonify({'error': 'Missing theme'}), 400

    try:
        theme = data['theme'].strip()

        haiku_text = generate_haiku(theme)
        if not haiku_text:
            return jsonify({'error': 'Generation failed'}), 500

        stats = count_syllables_and_words(haiku_text)

        logger.info(
            'haiku_service // generation ok, theme_len={}, words={}, syllables={}'.format(
                len(theme),
                stats['total_words'],
                stats['syllables_per_line'],
            )
        )

        return jsonify(
            {
                'haiku_text': haiku_text,
                'syllables_per_line': stats['syllables_per_line'],
                'total_words': stats['total_words'],
                'theme': theme,
            }
        )
    except Exception as ex:
        logger.error(f'haiku_service // generation: unexpected error: {ex}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for haiku service.
    """
    payload = {'status': 'ok'}
    logger.info('haiku_service // health status={}'.format(payload['status']))
    return jsonify(payload), 200


if __name__ == '__main__':
    app.run(
        host='localhost',
        port=config.tool_haiku_port,
        debug=config.flask_debug,
    )
