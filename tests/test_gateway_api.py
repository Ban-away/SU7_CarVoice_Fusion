from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz() -> None:
    response = client.get('/healthz')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_chat_task_http() -> None:
    response = client.post('/api/v1/chat', json={'message': '请导航到公司'})
    body = response.json()

    assert response.status_code == 200
    assert body['type'] == 'task_result'
    assert body['trace']['route'] == 'Task'
    assert body['session_id']


def test_chat_faq_http_with_citations() -> None:
    response = client.post('/api/v1/chat', json={'message': 'SU7 续航是多少'})
    body = response.json()

    assert response.status_code == 200
    assert body['type'] == 'faq_answer'
    assert len(body['citations']) > 0
    assert 'source' in body['citations'][0]
    assert 'page' in body['citations'][0]


def test_chat_unknown_http_clarification() -> None:
    response = client.post('/api/v1/chat', json={'message': 'abcdefg'})
    body = response.json()

    assert response.status_code == 200
    assert body['type'] == 'clarification'


def test_chat_rejects_blank_message() -> None:
    response = client.post('/api/v1/chat', json={'message': '   '})

    assert response.status_code == 422


def test_websocket_chat_flow() -> None:
    with client.websocket_connect('/ws/chat') as ws:
        ws.send_json({'message': '请播放音乐'})
        message = ws.receive_json()

    assert message['type'] == 'task_result'
    assert message['trace']['route'] == 'Task'


def test_high_risk_confirm_with_session() -> None:
    first = client.post('/api/v1/chat', json={'message': '请关闭安全系统'}).json()
    second = client.post(
        '/api/v1/chat',
        json={'message': '确认执行', 'confirm': True, 'session_id': first['session_id']},
    ).json()

    assert first['type'] == 'clarification'
    assert first['trace']['fallback_reason'] == 'high_risk_needs_confirmation'
    assert second['type'] == 'task_result'
    assert second['trace']['fallback_reason'] == 'confirmed_pending_skill'


def test_skills_endpoint_returns_whitelist_metadata() -> None:
    response = client.get('/api/v1/skills')
    payload = response.json()

    assert response.status_code == 200
    assert len(payload) >= 6
    assert {'name', 'risk_level', 'category', 'description', 'keywords'}.issubset(payload[0].keys())


def test_knowledge_retrieve_endpoint() -> None:
    response = client.post('/api/v1/knowledge/retrieve', json={'query': 'SU7 续航', 'top_k': 2})
    body = response.json()

    assert response.status_code == 200
    assert body['hit_count'] >= 1
    assert len(body['citations']) >= 1
