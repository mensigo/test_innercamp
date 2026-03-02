"""Prepare deterministic students CSV for API and future RAG steps."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import faiss
import numpy as np

from src.utils import post_embeddings

SUBJECTS = [
    'Machine Learning',
    'Probability Theory',
    'Optimization Theory',
]

STUDENTS = [
    'Алексеева Елена',
    'Андреев Матвей',
    'Васильев Артем',
    'Волкова Екатерина',
    'Ежик Ольга',
    'Емелин Илья',
    'Захаров Никита',
    'Иванова Анастасия',
    'Козлов Кирилл',
    'Козлова Татьяна',
    'Кузнецов Алексей',
    'Лебедев Иван',
    'Макаров Михаил',
    'Михайлов Даниил',
    'Морозова Евгения',
    'Никитина Наталья',
    'Николаев Максим',
    'Николаева Юлия',
    'Новикова Дарья',
    'Орлова Мария',
    'Павлова Ирина',
    'Павлова Светлана',
    'Петрушина Виктория',
    'Порогов Андрей',
    'Семенов Игорь',
    'Смирнов Александр',
    'Соколова Анна',
    'Степанов Тимофей',
    'Степанова Ксения',
    'Федоров Дмитрий',
]

DATA_DIR = Path(__file__).resolve().parent / 'data'
STUDENTS_CSV = DATA_DIR / 'students.csv'
FAISS_INDEX_PATH = DATA_DIR / 'faiss.index'
RAG_CHUNKS_PATH = DATA_DIR / 'rag_chunks.json'
DEFAULT_SEED = 42


def _subjects_for_student(student_idx: int, student_name: str) -> list[str]:
    """Return 1-2 subjects assigned to student."""
    if student_name == 'John':
        return ['Machine Learning']
    first = SUBJECTS[student_idx % len(SUBJECTS)]
    second = SUBJECTS[(student_idx + 1) % len(SUBJECTS)]
    return [first, second]


def _score_for(student_idx: int, subject_idx: int, seed: int) -> float:
    """Build deterministic seed-based score in range 3.0..5.0."""
    rng = random.Random(seed + student_idx * 101 + subject_idx * 37)
    score = rng.uniform(3.0, 5.0)
    return round(score, 1)


def build_students_rows(seed: int = DEFAULT_SEED) -> list[dict[str, str]]:
    """Build rows for students.csv with expected schema."""
    rows: list[dict[str, str]] = []
    for student_idx, student_name in enumerate(STUDENTS):
        for subject_name in _subjects_for_student(student_idx, student_name):
            subject_idx = SUBJECTS.index(subject_name)
            rows.append(
                {
                    'student_name': student_name,
                    'subject_name': subject_name,
                    'score': f'{_score_for(student_idx, subject_idx, seed):.1f}',
                }
            )
    return rows


def ensure_students_csv(force: bool = False, seed: int = DEFAULT_SEED) -> Path:
    """Create students.csv when missing or when force=True."""
    if STUDENTS_CSV.exists() and not force:
        return STUDENTS_CSV

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with STUDENTS_CSV.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=['student_name', 'subject_name', 'score'],
        )
        writer.writeheader()
        for row in build_students_rows(seed=seed):
            writer.writerow(row)
    return STUDENTS_CSV


def _load_markdown_chunks() -> list[str]:
    """Load markdown files from data directory as one chunk per file."""
    chunks: list[str] = []
    for file_path in sorted(DATA_DIR.glob('*.md')):
        text = file_path.read_text(encoding='utf-8').strip()
        if text:
            chunks.append(text)
    return chunks


def _extract_embeddings(response: dict) -> list[list[float]]:
    """Extract embeddings from API response ordered by source index."""
    raw_data = response['data']
    if not isinstance(raw_data, list):
        raise RuntimeError('Embeddings response has invalid data')

    embeddings = [
        item['embedding'] for item in sorted(raw_data, key=lambda x: x['index'])
    ]
    return embeddings


def build_faiss_index(force: bool = True) -> Path:
    """Build and persist FAISS index for markdown chunks."""
    if FAISS_INDEX_PATH.exists() and RAG_CHUNKS_PATH.exists() and not force:
        return FAISS_INDEX_PATH

    chunks = _load_markdown_chunks()
    if not chunks:
        raise RuntimeError('No markdown files found in src/data to build FAISS index')

    response = post_embeddings({'input': chunks}, verbose=True)
    embeddings = _extract_embeddings(response)
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f'Embeddings count mismatch: expected {len(chunks)}, got {len(embeddings)}'
        )

    vectors = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    RAG_CHUNKS_PATH.write_text(
        json.dumps({'chunks': chunks}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return FAISS_INDEX_PATH


if __name__ == '__main__':
    students_csv_path = ensure_students_csv(force=True, seed=DEFAULT_SEED)

    print(f'Сгенерирован файл со студентами: {students_csv_path}')
    with open(students_csv_path, encoding='utf-8') as f:
        rows = list(csv.reader(f))
        print(f'Всего строк: {len(rows) - 1}')
        print(f'Колонки: {", ".join(rows[0])}')

    faiss_index_path = build_faiss_index()

    print(f'\nСгенерирован FAISS-индекс: {faiss_index_path}')
    rag_chunks_file = Path(
        str(faiss_index_path).replace('faiss.index', 'rag_chunks.json')
    )
    data = json.loads(rag_chunks_file.read_text(encoding='utf-8'))
    print(f'Всего чанков: {len(data["chunks"])}')
    print('Первые 5 чанков:')
    for i, chunk in enumerate(data['chunks'][:5], 1):
        first_row = chunk.splitlines()[0] if chunk.splitlines() else ''
        print(f'{i}) {first_row if first_row.strip() else "<пусто>"}')
