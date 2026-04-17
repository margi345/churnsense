import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path
from src.data.loader import load_config, load_processed


# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


# ── Load model and features ────────────────────────────────────────────────────
def _load_model_and_features():
    model         = joblib.load(MODEL_DIR / "best_model.joblib")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    return model, feature_names


# ── Build SHAP explainer ───────────────────────────────────────────────────────
def build_explainer(model, X: pd.DataFrame):
    print("Building SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    print("SHAP explainer ready.")
    return explainer


# ── Global explanation: which features matter most overall ────────────────────
def explain_global(top_n: int = 15) -> pd.DataFrame:
    model, feature_names = _load_model_and_features()
    df = load_processed()

    # Remove target column
    X = df[[c for c in feature_names if c in df.columns]]

    explainer  = build_explainer(model, X)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP value per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": mean_shap
    }).sort_values("importance", ascending=False).head(top_n)

    print(f"\nTop {top_n} most important features globally:")
    print(importance_df.to_string(index=False))

    # Save for dashboard use
    importance_df.to_parquet(MODEL_DIR / "global_shap.parquet", index=False)

    return importance_df


# ── Local explanation: why THIS customer is at risk ───────────────────────────
def explain_local(customer_features: pd.DataFrame, top_n: int = 5) -> list:
    """
    Returns top N risk factors for a single customer.

    Args:
        customer_features: single-row DataFrame with model features
        top_n: number of top factors to return

    Returns:
        list of dicts like:
        [{"feature": "is_month_to_month", "direction": "increases", "magnitude": 0.23}, ...]
    """
    model, feature_names = _load_model_and_features()

    # Align columns
    for col in feature_names:
        if col not in customer_features.columns:
            customer_features[col] = 0
    X = customer_features[[c for c in feature_names if c in customer_features.columns]]

    explainer   = build_explainer(model, X)
    shap_values = explainer.shap_values(X)

    # Build explanation
    factors = []
    for i, feature in enumerate(feature_names):
        if feature not in X.columns:
            continue
        shap_val = float(shap_values[0][i])
        factors.append({
            "feature":   feature,
            "shap_value": shap_val,
            "direction": "increases churn risk" if shap_val > 0 else "decreases churn risk",
            "magnitude": round(abs(shap_val), 4),
            "value":     float(X[feature].values[0])
        })

    # Sort by absolute magnitude
    factors = sorted(factors, key=lambda x: x["magnitude"], reverse=True)[:top_n]

    return factors


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Global explanation
    importance = explain_global(top_n=10)

    # Local explanation for a test customer
    print("\nLocal explanation for high-risk customer:")
    test = pd.DataFrame([{
        "tenure":              2,
        "MonthlyCharges":      85.0,
        "TotalCharges":        170.0,
        "is_month_to_month":   1,
        "is_new_customer":     1,
        "total_services":      2,
        "churn_risk_score":    7,
        "has_fiber":           1,
        "has_security_services": 0,
        "is_auto_payment":     0,
    }])

    factors = explain_local(test, top_n=5)
    print("\nTop 5 risk factors:")
    for i, f in enumerate(factors, 1):
        print(f"  {i}. {f['feature']}: {f['direction']} (magnitude: {f['magnitude']})")