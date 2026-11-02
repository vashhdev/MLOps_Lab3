import pytest
from fastapi.testclient import TestClient

import predict
from main import app


@pytest.fixture(scope="module")
def client():
    # Using the client as a context manager runs the app's startup (model loading).
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_urgent(client):
    response = client.post("/predict", json={"message": "my father collapsed and isn't responding"})
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "urgent"
    assert 0 < body["confidence"] <= 1
    assert set(body["all_probabilities"]) == {"urgent", "normal", "low"}
    assert body["confidence"] == max(body["all_probabilities"].values())


def test_predict_low(client):
    response = client.post("/predict", json={"message": "what are your office hours on Saturday"})
    assert response.status_code == 200
    assert response.json()["urgency"] == "low"


def test_predict_batch(client):
    messages = [
        "my father collapsed and isn't responding",
        "I need to refill my blood pressure prescription",
        "what are your office hours on Saturday",
    ]
    response = client.post("/predict/batch", json={"messages": messages})
    assert response.status_code == 200
    results = response.json()
    assert [r["urgency"] for r in results] == ["urgent", "normal", "low"]


def test_too_short_message_rejected(client):
    response = client.post("/predict", json={"message": "hi"})
    assert response.status_code == 422


def test_whitespace_only_message_rejected(client):
    response = client.post("/predict", json={"message": "     "})
    assert response.status_code == 422
    assert "non-whitespace" in response.json()["detail"]


def test_batch_over_limit_rejected(client):
    response = client.post("/predict/batch", json={"messages": ["where do I park"] * 51})
    assert response.status_code == 422


def test_missing_model_returns_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "_model", None)
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "missing.pkl")
    assert client.post("/predict", json={"message": "where do I park"}).status_code == 503
    assert client.get("/health").status_code == 503
