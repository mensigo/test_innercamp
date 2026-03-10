"""Student meta-grade tool."""

from __future__ import annotations

from src_example.data_store import (
    load_meta_grades,
    normalize_student_name,
    normalize_subject_name,
)


def student_meta_tool(student_name: str, subject_name: str) -> dict:
    """Return meta-grades and failed question ids for student+subject."""
    normalized_student = normalize_student_name(student_name)
    normalized_subject = normalize_subject_name(subject_name)
    if normalized_student is None:
        return {'error': f'Unknown student: {student_name}'}
    if normalized_subject is None:
        return {'error': f'Unknown subject: {subject_name}'}

    grades_map = load_meta_grades().get((normalized_student, normalized_subject))
    if not grades_map:
        return {
            'error': f'No grades for student={normalized_student}, subject={normalized_subject}'
        }

    failed_questions = sorted(
        question_id for question_id, grade in grades_map.items() if grade < 50
    )
    sorted_grades = {
        str(question_id): grades_map[question_id] for question_id in sorted(grades_map)
    }

    return {
        'student_name': normalized_student,
        'subject_name': normalized_subject,
        'failed_questions': failed_questions,
        'meta_grades': sorted_grades,
    }
