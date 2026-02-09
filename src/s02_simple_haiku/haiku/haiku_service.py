"""Flask service for haiku generation."""

from flask import Flask, jsonify, request

from src import config, post_chat_completions

from .split_word import split_into_syllables_simple

app = Flask(__name__)


def generate_haiku(topic: str, **kwargs) -> str:
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

    user_prompt = f'Напиши хайку на тему: {topic}'

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': kwargs.get('temperature', 0.5),
    }

    response = post_chat_completions(payload)

    if 'error' in response:
        print(f'LLM Error: {response["error"]}')
        return ''

    try:
        content = response['choices'][0]['message']['content']
    except (KeyError, IndexError) as exc:
        print(f'LLM Missing expected key/index ({exc}): {response}')
        return ''

    if not isinstance(content, str) or not content.strip():
        print(f'LLM Missing content in response: {response}')
        return ''

    haiku = content.strip()
    return haiku


def count_syllables_and_words(haiku_text: str) -> dict:
    """
    Count syllables per line and total words in haiku text.
    """
    lines = haiku_text.strip().split('\n')
    syllables_per_line = []
    total_words = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        words = line.split()
        line_syllable_count = 0

        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word:
                syllables = split_into_syllables_simple(clean_word)
                line_syllable_count += len(syllables)
                total_words += 1

        syllables_per_line.append(line_syllable_count)

    return {
        'syllables_per_line': syllables_per_line,
        'total_words': total_words,
    }


@app.route('/generate_haiku', methods=['POST'])
def generate_haiku_endpoint():
    """
    Generate haiku on given topic and return with syllable stats.
    """
    data = request.get_json()

    if not data or 'topic' not in data:
        return jsonify({'error': 'Missing topic field'}), 400

    topic = data['topic']

    haiku_text = generate_haiku(topic)
    if not haiku_text:
        return jsonify({'error': 'Failed to generate haiku'}), 500

    stats = count_syllables_and_words(haiku_text)

    return jsonify(
        {
            'haiku_text': haiku_text,
            'syllables_per_line': stats['syllables_per_line'],
            'total_words': stats['total_words'],
            'topic': topic,
        }
    )


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for haiku service.
    """
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    app.run(
        host='localhost',
        port=config.tool_haiku_port,
        debug=config.flask_debug,
    )
