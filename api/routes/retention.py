from fastapi import APIRouter, HTTPException
from api.schemas import CustomerRequest, RetentionPlanResponse
from src.models.predict import predict
from src.explainability.shap_explainer import explain_local
from src.retention.llm_engine import RetentionEngine
from src.data.cleaner import clean
from src.data.features import build_features
import pandas as pd

router = APIRouter()
engine = RetentionEngine()


@router.post("/retention-plan", response_model=RetentionPlanResponse)
def get_retention_plan(customer: CustomerRequest):
    try:
        customer_dict = customer.model_dump()

        # Get prediction
        prediction = predict(customer_dict)
        customer_dict.update(prediction)

        # Get SHAP factors
        df      = pd.DataFrame([customer_dict])
        df      = clean(df)
        df      = build_features(df)
        factors = explain_local(df, top_n=5)

        # Generate LLM retention plan
        plan = engine.generate(customer_dict, factors)

        return RetentionPlanResponse(**plan)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))