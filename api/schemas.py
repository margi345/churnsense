from pydantic import BaseModel, Field
from typing import Optional


# ── Request schema ─────────────────────────────────────────────────────────────
class CustomerRequest(BaseModel):
    tenure:           float = Field(..., ge=0, description="Months with company")
    MonthlyCharges:   float = Field(..., ge=0, description="Current monthly bill")
    TotalCharges:     float = Field(..., ge=0, description="Total billed to date")
    Contract:         str   = Field(..., description="Month-to-month, One year, Two year")
    PaymentMethod:    str   = Field(..., description="Payment method")
    InternetService:  str   = Field(..., description="DSL, Fiber optic, No")
    OnlineSecurity:   str   = Field("No", description="Yes, No, No internet service")
    TechSupport:      str   = Field("No", description="Yes, No, No internet service")
    StreamingTV:      str   = Field("No", description="Yes, No, No internet service")
    StreamingMovies:  str   = Field("No", description="Yes, No, No internet service")
    PhoneService:     str   = Field("Yes", description="Yes, No")
    MultipleLines:    str   = Field("No", description="Yes, No, No phone service")
    OnlineBackup:     str   = Field("No", description="Yes, No, No internet service")
    DeviceProtection: str   = Field("No", description="Yes, No, No internet service")
    PaperlessBilling: str   = Field("Yes", description="Yes, No")
    Partner:          str   = Field("No", description="Yes, No")
    Dependents:       str   = Field("No", description="Yes, No")
    SeniorCitizen:    int   = Field(0, ge=0, le=1, description="0 or 1")
    gender:           str   = Field("Male", description="Male, Female")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure":           2,
                "MonthlyCharges":   85.0,
                "TotalCharges":     170.0,
                "Contract":         "Month-to-month",
                "PaymentMethod":    "Electronic check",
                "InternetService":  "Fiber optic",
                "OnlineSecurity":   "No",
                "TechSupport":      "No",
                "StreamingTV":      "Yes",
                "StreamingMovies":  "Yes",
                "PhoneService":     "Yes",
                "MultipleLines":    "No",
                "OnlineBackup":     "No",
                "DeviceProtection": "No",
                "PaperlessBilling": "Yes",
                "Partner":          "No",
                "Dependents":       "No",
                "SeniorCitizen":    0,
                "gender":           "Male"
            }
        }


# ── Prediction response ────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_score:      float
    churn_percent:    str
    risk_tier:        str
    survival_day_30:  Optional[float] = None
    survival_day_60:  Optional[float] = None
    survival_day_90:  Optional[float] = None
    churn_prob_day_30: Optional[float] = None
    churn_prob_day_60: Optional[float] = None
    churn_prob_day_90: Optional[float] = None


# ── Explanation response ───────────────────────────────────────────────────────
class RiskFactor(BaseModel):
    feature:   str
    direction: str
    magnitude: float
    value:     float

class ExplanationResponse(BaseModel):
    churn_score:  float
    risk_tier:    str
    top_factors:  list[RiskFactor]


# ── Retention action ───────────────────────────────────────────────────────────
class RetentionAction(BaseModel):
    type:            str
    title:           str
    message:         str
    offer:           str
    expected_impact: str

class RetentionPlanResponse(BaseModel):
    segment:  str
    urgency:  str
    summary:  str
    actions:  list[RetentionAction]
    source:   Optional[str] = "llm"


# ── Health check ───────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:  str
    version: str