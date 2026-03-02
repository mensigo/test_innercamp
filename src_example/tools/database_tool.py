"""Database-like analytics tool over local deterministic CSV data."""

from __future__ import annotations

from src_example.data_store import (
    SUBJECTS,
    load_enrollments,
    load_meta_grades,
    normalize_subject_name,
)


def _final_grade(student_name: str, subject_name: str) -> float | None:
    """Compute final grade for student in subject."""
    grades_map = load_meta_grades().get((student_name, subject_name))
    if not grades_map:
        return None
    values = sorted(grades_map.items(), key=lambda item: item[0])
    return round(sum(grade for _, grade in values) / len(values), 1)


def _subject_students(subject_name: str) -> list[str]:
    """Get students enrolled in selected subject."""
    enrollments = load_enrollments()
    students = [
        student_name
        for student_name, subjects in enrollments.items()
        if subject_name in subjects
    ]
    return sorted(students)


def _top_students(subject_name: str, top_k: int) -> list[dict]:
    """Return top students by final grade for subject."""
    rows: list[dict] = []
    for student_name in _subject_students(subject_name):
        grade = _final_grade(student_name, subject_name)
        if grade is None:
            continue
        rows.append({'name': student_name, 'grade': grade})
    rows.sort(key=lambda row: (-row['grade'], row['name']))
    return rows[:top_k]


def _hardest_questions(subject_name: str, top_k: int) -> list[dict]:
    """Return hardest questions by lowest average meta-grade."""
    meta = load_meta_grades()
    question_grades: dict[int, list[int]] = {}
    for (student_name, sub_name), grades_map in meta.items():
        if sub_name != subject_name:
            continue
        for question_id, grade in grades_map.items():
            question_grades.setdefault(question_id, []).append(grade)

    rows: list[dict] = []
    for question_id in sorted(question_grades):
        grades = question_grades[question_id]
        avg_grade = round(sum(grades) / len(grades), 1)
        rows.append({'question_id': question_id, 'avg_grade': avg_grade})
    rows.sort(key=lambda row: (row['avg_grade'], row['question_id']))
    return rows[:top_k]


def _failure_stats(subject_name: str, failed_share_threshold: float) -> int:
    """Count questions failed by more than threshold of enrolled students."""
    students = _subject_students(subject_name)
    if not students:
        return 0

    meta = load_meta_grades()
    failed_counts: dict[int, int] = {question_id: 0 for question_id in range(1, 11)}
    total = len(students)

    for student_name in students:
        grades_map = meta.get((student_name, subject_name)) or {}
        for question_id in range(1, 11):
            if grades_map.get(question_id, 0) < 50:
                failed_counts[question_id] += 1

    count = 0
    for question_id in range(1, 11):
        share = failed_counts[question_id] / total
        if share > failed_share_threshold:
            count += 1
    return count


def database_tool(
    operation: str,
    subject_name: str,
    top_k: int = 3,
    failed_share_threshold: float = 0.8,
) -> dict:
    """Run deterministic analytics operation over local grade data."""
    normalized_subject = normalize_subject_name(subject_name)
    if normalized_subject is None:
        return {'error': f'Unknown subject: {subject_name}', 'data': None}

    if normalized_subject not in SUBJECTS:
        return {'error': f'Unsupported subject: {normalized_subject}', 'data': None}

    safe_top_k = max(1, min(int(top_k), 30))
    if operation == 'top_students':
        return {
            'operation': operation,
            'subject_name': normalized_subject,
            'data': _top_students(normalized_subject, safe_top_k),
        }

    if operation == 'hardest_questions':
        return {
            'operation': operation,
            'subject_name': normalized_subject,
            'data': _hardest_questions(normalized_subject, safe_top_k),
        }

    if operation == 'failure_stats':
        return {
            'operation': operation,
            'subject_name': normalized_subject,
            'data': {
                'count': _failure_stats(normalized_subject, failed_share_threshold)
            },
        }

    return {'error': f'Unsupported operation: {operation}', 'data': None}
