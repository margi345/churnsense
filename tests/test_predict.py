import pytest
from src.models.predict import predict


@pytest.fixture
def high_risk_customer():
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


@pytest.fixture
def low_risk_customer():
    return {
        "tenure": 60,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
        "Contract": "Two year",
        "PaymentMethod": "Bank transfer (automatic)",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "PaperlessBilling": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "SeniorCitizen": 0,
        "gender": "Female"
    }


def test_predict_returns_dict(high_risk_customer):
    result = predict(high_risk_customer)
    assert isinstance(result, dict)


def test_predict_has_required_keys(high_risk_customer):
    result = predict(high_risk_customer)
    assert "churn_score" in result
    assert "churn_percent" in result
    assert "risk_tier" in result


def test_predict_score_range(high_risk_customer):
    result = predict(high_risk_customer)
    assert 0.0 <= result["churn_score"] <= 1.0


def test_high_risk_customer_is_critical(high_risk_customer):
    result = predict(high_risk_customer)
    assert result["churn_score"] > 0.5
    assert result["risk_tier"] in ["high", "critical"]


def test_low_risk_customer_is_safe(low_risk_customer):
    result = predict(low_risk_customer)
    assert result["churn_score"] < 0.5
    assert result["risk_tier"] in ["low", "medium"]


def test_risk_tier_valid_values(high_risk_customer):
    result = predict(high_risk_customer)
    assert result["risk_tier"] in ["low", "medium", "high", "critical"]