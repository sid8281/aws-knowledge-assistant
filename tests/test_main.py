import pytest
from fastapi.testclient import TestClient

from app.config import settings


@pytest.fixture
def client(mocker):
    # Mock the heavy pipeline pieces before importing the app, so the
    # test doesn't touch Qdrant, the embedding model, or a real LLM.
    mocker.patch("app.main.check_input", return_value=(True, None))
    mocker.patch("app.main.check_output", return_value=(True, "Mocked answer."))
    mocker.patch(
        "app.main.run_query",
        return_value={
            "answer": "Mocked answer.",
            "steps": [{"node": "responder", "output": "ok"}],
            "retrieved_chunks": [{"source": "doc.json"}],
        },
    )

    from app.main import app

    return TestClient(app)


def test_query_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret-key")

    response = client.post("/query", json={"question": "What is S3?"})

    assert response.status_code == 401


def test_query_succeeds_with_valid_api_key(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret-key")

    response = client.post(
        "/query",
        json={"question": "What is S3?"},
        headers={"X-API-Key": "secret-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Mocked answer."
    assert body["sources"] == ["doc.json"]
    assert body["thread_id"]  # server generated one


def test_query_auth_disabled_when_no_api_key_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")

    response = client.post("/query", json={"question": "What is S3?"})

    assert response.status_code == 200


def test_query_preserves_client_thread_id(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")

    response = client.post(
        "/query",
        json={"question": "What is S3?", "thread_id": "my-thread"},
    )

    assert response.json()["thread_id"] == "my-thread"


def test_query_rejects_empty_question(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")

    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_query_rejects_oversized_question(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")

    response = client.post("/query", json={"question": "x" * 3000})

    assert response.status_code == 422


def test_health_returns_status(client):
    response = client.get("/health")

    assert response.status_code in (200, 503)
    assert "checks" in response.json()
