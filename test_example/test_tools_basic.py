"""Basic happy-path tests for local tools."""

from src_example.tools.database_tool import database_tool
from src_example.tools.rag_tool import rag_tool
from src_example.tools.student_meta_tool import student_meta_tool


def test_database_tool_top_students_basic():
    result = database_tool(
        operation='top_students',
        subject_name='Machine Learning',
        top_k=5,
    )
    assert 'error' not in result
    assert result['operation'] == 'top_students'
    assert len(result['data']) == 5
    assert result['data'][0]['grade'] >= result['data'][-1]['grade']


def test_student_meta_tool_basic():
    result = student_meta_tool(student_name='John', subject_name='Machine Learning')
    assert 'error' not in result
    assert result['student_name'] == 'John'
    assert result['subject_name'] == 'Machine Learning'
    assert isinstance(result['failed_questions'], list)
    assert '1' in result['meta_grades']


def test_rag_tool_basic():
    result = rag_tool('Назовите 4 ключевые особенности машинного обучения.')
    assert 'error' not in result
    assert 'обобщающая способность' in result['answer'].lower()
