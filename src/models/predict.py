import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.data.loader import load_config
from src.data.features import build_features
from src.data.cleaner import clean


# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def load_model():
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Run training first: python -m src.models.train"
        )
    return joblib.load(model_path)


def load_feature_names() -> list:
    feature_path = MODEL_DIR / "feature_names.joblib"
    if not feature_path.exists():
        raise FileNotFoundError(
            "Feature names not found. Run training first: python -m src.models.train"
        )
    return joblib.load(feature_path)


def predict(customer: dict) -> dict:
    model         = load_model()
    feature_names = load_feature_names()

    df = pd.DataFrame([customer])
    df = clean(df)
    df = build_features(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[[col for col in feature_names if col in df.columns]]

    churn_prob = float(model.predict_proba(df)[0][1])

    if churn_prob >= 0.70:
        risk_tier = "critical"
    elif churn_prob >= 0.45:
        risk_tier = "high"
    elif churn_prob >= 0.25:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    return {
        "churn_score":   round(churn_prob, 4),
        "churn_percent": f"{churn_prob:.1%}",
        "risk_tier":     risk_tier,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    model         = load_model()
    feature_names = load_feature_names()

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df_model = df[[col for col in feature_names if col in df.columns]]

    probs = model.predict_proba(df_model)[:, 1]

    df = df.copy()
    df["churn_score"] = probs.round(4)
    df["risk_tier"] = pd.cut(
        probs,
        bins=[-np.inf, 0.25, 0.45, 0.70, np.inf],
        labels=["low", "medium", "high", "critical"]
    )

    return df


if __name__ == "__main__":
    test_customer = {
        "tenure":             2,
        "MonthlyCharges":     85.0,
        "TotalCharges":       170.0,
        "Contract":           "Month-to-month",
        "PaymentMethod":      "Electronic check",
        "InternetService":    "Fiber optic",
        "OnlineSecurity":     "No",
        "TechSupport":        "No",
        "StreamingTV":        "Yes",
        "StreamingMovies":    "Yes",
        "PhoneService":       "Yes",
        "MultipleLines":      "No",
        "OnlineBackup":       "No",
        "DeviceProtection":   "No",
        "PaperlessBilling":   "Yes",
        "Partner":            "No",
        "Dependents":         "No",
        "SeniorCitizen":      0,
        "gender":             "Male",
    }

    result = predict(test_customer)
    print("\nTest Customer Prediction:")
    print(f"  Churn Score:  {result['churn_score']}")
    print(f"  Churn Risk:   {result['churn_percent']}")
    print(f"  Risk Tier:    {result['risk_tier'].upper()}")
    