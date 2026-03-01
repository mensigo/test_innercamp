"""Initial simple tests for src_learn agent."""

import json

from src_learn.agent import IRRELEVANT_MESSAGE, agent


def test_agent_irrelevant_query(monkeypatch):
    calls = {'count': 0}

    def fake_post_chat_completions(payload: dict, verbose: bool = False) -> dict:
        calls['count'] += 1
        return {'choices': [{'message': {'content': 'irrelevant'}}]}

    monkeypatch.setattr(
        'src_learn.classify_intent.post_chat_completions',
        fake_post_chat_completions,
    )

    result = agent([{'role': 'user', 'content': 'Tell me a football joke'}])
    assert result == {'answer': IRRELEVANT_MESSAGE}
    assert calls['count'] == 1


def test_agent_top_students_happy_path(monkeypatch):
    responses = [
        {'choices': [{'message': {'content': 'relevant'}}]},
        {
            'choices': [
                {
                    'message': {
                        'tool_calls': [
                            {
                                'function': {
                                    'name': 'route_query',
                                    'arguments': json.dumps(
                                        {
                                            'tool_name': 'database_tool',
                                            'operation': 'top_students',
                                            'subject_name': 'Machine Learning',
                                            'top_k': 3,
                                        }
                                    ),
                                }
                            }
                        ]
                    }
                }
            ]
        },
    ]

    def fake_classify_post_chat_completions(
        payload: dict, verbose: bool = False
    ) -> dict:
        return responses.pop(0)

    def fake_route_post_chat_completions(payload: dict, verbose: bool = False) -> dict:
        return responses.pop(0)

    monkeypatch.setattr(
        'src_learn.classify_intent.post_chat_completions',
        fake_classify_post_chat_completions,
    )
    monkeypatch.setattr(
        'src_learn.router.post_chat_completions',
        fake_route_post_chat_completions,
    )

    result = agent([{'role': 'user', 'content': 'Top-3 students in Machine Learning'}])
    assert 'answer' in result
    answer = result['answer']
    assert isinstance(answer, str) and answer
    assert ', ' in answer
    assert '(' in answer and ')' in answer
