from fastapi import APIRouter, HTTPException
from api.schemas import CustomerRequest, ExplanationResponse
from src.models.predict import predict
from src.explainability.shap_explainer import explain_local
from src.data.cleaner import clean
from src.data.features import build_features
import pandas as pd

router = APIRouter()


@router.post("/explain", response_model=ExplanationResponse)
def explain_churn(customer: CustomerRequest):
    try:
        customer_dict = customer.model_dump()

        # Get churn score
        prediction = predict(customer_dict)

        # Build features for SHAP
        df = pd.DataFrame([customer_dict])
        df = clean(df)
        df = build_features(df)

        # Get SHAP explanation
        factors = explain_local(df, top_n=5)

        return ExplanationResponse(
            churn_score  = prediction["churn_score"],
            risk_tier    = prediction["risk_tier"],
            top_factors  = factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))