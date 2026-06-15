import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data

def test_predict_endpoint_success():
    response = client.post(
        "/predict",
        json={"symptoms": ["itching", "skin rash"], "model_type": "xgboost"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_disease" in data
    assert "confidence" in data
    assert "risk_assessment" in data
    assert "symptoms_matched" in data
    assert "itching" in data["symptoms_matched"]

def test_predict_endpoint_no_match():
    response = client.post(
        "/predict",
        json={"symptoms": ["invalid symptom name"], "model_type": "xgboost"}
    )
    assert response.status_code == 400

def test_predict_topk_endpoint():
    response = client.post(
        "/predict_topk",
        json={"symptoms": ["itching", "skin rash"], "k": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3

def test_chat_endpoint_conversational():
    response = client.post(
        "/chat",
        json={"user_text": "hello there!", "session_id": "test_session"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["type"] == "chat"

def test_chat_endpoint_diagnosis():
    response = client.post(
        "/chat",
        json={"user_text": "I have a headache and muscle pain", "session_id": "test_session"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["type"] == "diagnosis"
    assert len(data["symptoms_found"]) > 0
