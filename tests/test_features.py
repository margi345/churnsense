import pytest
import pandas as pd
import numpy as np
from src.data.cleaner import clean
from src.data.features import build_features


@pytest.fixture
def sample_df():
    return pd.DataFrame([{
        "tenure": 12,
        "MonthlyCharges": 65.0,
        "TotalCharges": "780.0",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "PaperlessBilling": "Yes",
        "Partner": "Yes",
        "Dependents": "No",
        "SeniorCitizen": 0,
        "gender": "Female",
        "Churn": "No"
    }])


def test_clean_no_nulls(sample_df):
    result = clean(sample_df)
    assert result.isnull().sum().sum() == 0


def test_clean_drops_customer_id(sample_df):
    result = clean(sample_df)
    assert "customerID" not in result.columns


def test_clean_churn_is_numeric(sample_df):
    result = clean(sample_df)
    assert result["Churn"].dtype in [int, float, np.int64, np.float64]


def test_build_features_adds_columns(sample_df):
    cleaned  = clean(sample_df)
    features = build_features(cleaned)
    assert "is_month_to_month" in features.columns
    assert "churn_risk_score" in features.columns
    assert "total_services" in features.columns
    assert "tenure_years" in features.columns


def test_build_features_no_nulls(sample_df):
    cleaned  = clean(sample_df)
    features = build_features(cleaned)
    assert features.isnull().sum().sum() == 0


def test_churn_risk_score_range(sample_df):
    cleaned  = clean(sample_df)
    features = build_features(cleaned)
    assert features["churn_risk_score"].min() >= 0
    assert features["churn_risk_score"].max() <= 10