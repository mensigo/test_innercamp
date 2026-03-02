"""RAG-style lookup tool over local exam markdown files."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from src_example.data_store import EXAMS_DIR, ensure_data_files


def _tokenize(text: str) -> set[str]:
    """Tokenize text for deterministic overlap scoring."""
    return set(re.findall(r'[a-zа-яё0-9]+', text.lower()))


@lru_cache(maxsize=1)
def _load_exam_entries() -> list[dict]:
    """Load exam Q/A entries from markdown files."""
    ensure_data_files()
    entries: list[dict] = []
    for path in sorted(Path(EXAMS_DIR).glob('*.md')):
        subject_name = path.stem.replace('_', ' ').title()
        content = path.read_text(encoding='utf-8')
        pattern = re.compile(
            r'## Q(?P<qid>\d+): (?P<question>.+?)\nA:\n(?P<answer>.*?)(?=\n## Q|\Z)',
            flags=re.S,
        )
        for match in pattern.finditer(content):
            question_id = int(match.group('qid'))
            question = match.group('question').strip()
            answer = match.group('answer').strip()
            entries.append(
                {
                    'subject_name': subject_name,
                    'question_id': question_id,
                    'question': question,
                    'answer': answer,
                }
            )
    return entries


def rag_tool(question: str) -> dict:
    """Return deterministic answer from markdown corpus."""
    query = question.strip()
    if not query:
        return {'error': 'Empty question'}

    query_tokens = _tokenize(query)
    scored: list[tuple[int, int, str, dict]] = []
    for entry in _load_exam_entries():
        question_tokens = _tokenize(entry['question'])
        overlap = len(query_tokens.intersection(question_tokens))
        scored.append(
            (
                overlap,
                entry['question_id'],
                entry['subject_name'],
                entry,
            )
        )

    scored.sort(key=lambda row: (-row[0], row[2], row[1]))
    best = scored[0][3]
    if scored[0][0] == 0:
        return {
            'question': query,
            'answer': 'No relevant exam answer found in local markdown corpus.',
            'source': {
                'subject_name': best['subject_name'],
                'question_id': best['question_id'],
                'question': best['question'],
            },
        }

    return {
        'question': query,
        'answer': best['answer'],
        'source': {
            'subject_name': best['subject_name'],
            'question_id': best['question_id'],
            'question': best['question'],
        },
    }
