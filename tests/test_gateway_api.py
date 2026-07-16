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
