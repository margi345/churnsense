from fastapi import APIRouter, HTTPException
from api.schemas import CustomerRequest, PredictionResponse
import pandas as pd
import joblib
from pathlib import Path

router = APIRouter()

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def _predict(customer_dict: dict) -> dict:
    model         = joblib.load(MODEL_DIR / "best_model.joblib")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")

    from src.data.cleaner import clean
    from src.data.features import build_features

    df = pd.DataFrame([customer_dict])
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


@router.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerRequest):
    try:
        customer_dict = customer.model_dump()
        result = _predict(customer_dict)

        # Survival probabilities
        try:
            survival_path = MODEL_DIR / "survival_model.joblib"
            if survival_path.exists():
                from src.models.survival import SurvivalModel
                survival_model = joblib.load(survival_path)

                from src.data.cleaner import clean
                from src.data.features import build_features
                df = pd.DataFrame([customer_dict])
                df = clean(df)
                df = build_features(df)

                for col in survival_model.feature_cols:
                    if col not in df.columns:
                        df[col] = 0
                df_s = df[survival_model.feature_cols]
                survival = survival_model.predict_survival_at_days(df_s)
                result.update(survival)
        except Exception as e:
            print(f"Survival error: {e}")

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))