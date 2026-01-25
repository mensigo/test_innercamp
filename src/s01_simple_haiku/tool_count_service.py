"""Flask service for counting syllables in haiku text."""

from flask import Flask, request, jsonify
from split_word import split_into_syllables_simple

app = Flask(__name__)


@app.route('/count', methods=['POST'])
def count_syllables():
    """
    Count syllables in provided text.
    Expects JSON with 'text' field containing haiku.
    Returns syllable counts per line and totals.
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    text = data['text']
    lines = text.strip().split('\n')

    syllables_per_line = []
    total_words = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Split line into words and count syllables
        words = line.split()
        line_syllable_count = 0

        for word in words:
            # Remove punctuation for syllable counting
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word:
                syllables = split_into_syllables_simple(clean_word)
                line_syllable_count += len(syllables)
                total_words += 1

        syllables_per_line.append(line_syllable_count)

    total_syllables = sum(syllables_per_line)

    return jsonify(
        {
            'syllables_per_line': syllables_per_line,
            'total_syllables': total_syllables,
            'total_words': total_words,
            'input_text': text,
        }
    )


if __name__ == '__main__':
    app.run(host='localhost', port=8090, debug=True)
