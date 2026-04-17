import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


@pytest.fixture
def sample_customer():
    return {
        "tenure": 2,
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "Dependents": "No",
        "SeniorCitizen": 0,
        "gender": "Male"
    }


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "ChurnSense" in response.json()["message"]


def test_predict_endpoint(sample_customer):
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert "churn_score" in data
    assert "risk_tier" in data
    assert 0.0 <= data["churn_score"] <= 1.0


def test_explain_endpoint(sample_customer):
    response = client.post("/explain", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert "top_factors" in data
    assert len(data["top_factors"]) > 0


def test_predict_missing_field():
    response = client.post("/predict", json={"tenure": 2})
    assert response.status_code == 422


def test_predict_invalid_tenure():
    bad_customer = {
        "tenure": -1,
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "Dependents": "No",
        "SeniorCitizen": 0,
        "gender": "Male"
    }
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422