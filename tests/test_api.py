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
    assert "top_3_predictions" in data
    assert len(data["top_3_predictions"]) == 3
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

def test_analyze_report_pdf():
    pdf_content = (
        b'%PDF-1.4\n'
        b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n'
        b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n'
        b'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> /Contents 4 0 R >>\nendobj\n'
        b'4 0 obj\n<< /Length 100 >>\nstream\n'
        b'BT\n/F1 12 Tf\n72 712 Td\n(Fasting Glucose: 112 mg/dL. Hemoglobin: 11.5 g/dL. Total Cholesterol: 210 mg/dL) Tj\nET\n'
        b'endstream\nendobj\n'
        b'xref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000056 00000 n\n0000000111 00000 n\n0000000241 00000 n\n'
        b'trailer\n<< /Size 5 /Root 1 0 R >>\n'
        b'startxref\n392\n%%EOF\n'
    )
    import io
    response = client.post(
        "/analyze_report",
        files={"file": ("report.pdf", io.BytesIO(pdf_content), "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "biomarkers" in data
    assert data["biomarkers"]["glucose"] == 112.0
    assert data["biomarkers"]["hemoglobin"] == 11.5
    assert data["biomarkers"]["cholesterol"] == 210.0
    assert "findings" in data
