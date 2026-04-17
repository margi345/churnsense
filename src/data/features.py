import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import load_config, load_interim


# Main feature engineering function 
def build_features(df: pd.DataFrame = None) -> pd.DataFrame:

    if df is None:
        df = load_interim()

    print("Building features...")
    df = df.copy()

    # Block 1: Tenure features 
    # How long has the customer been with us?
    df["tenure_years"] = df["tenure"] / 12
    df["is_new_customer"] = (df["tenure"] <= 3).astype(int)
    df["is_loyal_customer"] = (df["tenure"] >= 24).astype(int)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(int)

    # Block 2: Charge features 
    # Financial signals are strong churn predictors
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    df["charge_increase_ratio"] = np.where(
        df["avg_monthly_spend"] > 0,
        df["MonthlyCharges"] / df["avg_monthly_spend"],
        1.0
    )
    df["is_high_value"] = (
        df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)
    ).astype(int)
    df["is_low_value"] = (
        df["MonthlyCharges"] < df["MonthlyCharges"].quantile(0.25)
    ).astype(int)
    df["monthly_charges_log"] = np.log1p(df["MonthlyCharges"])
    df["total_charges_log"] = np.log1p(df["TotalCharges"])

    # Block 3: Service count features 
    # More services = more sticky = less likely to churn
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    # Convert service columns to binary first
    for col in service_cols:
        if df[col].dtype == object:
            df[col] = df[col].map({
                "Yes": 1, "No": 0,
                "No internet service": 0,
                "No phone service": 0
            })

    df["total_services"] = df[service_cols].sum(axis=1)
    df["has_streaming"] = (
        (df["StreamingTV"] == 1) | (df["StreamingMovies"] == 1)
    ).astype(int)
    df["has_security_services"] = (
        (df["OnlineSecurity"] == 1) | (df["DeviceProtection"] == 1)
    ).astype(int)
    df["has_support_services"] = (
        (df["TechSupport"] == 1) | (df["OnlineBackup"] == 1)
    ).astype(int)
    df["service_adoption_rate"] = df["total_services"] / len(service_cols)

    # Block 4: Contract and payment features
    # Month-to-month contracts churn much more
    df["is_month_to_month"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)
    df["is_long_term_contract"] = (
        df["Contract"].isin(["One year", "Two year"])
    ).astype(int)
    df["is_auto_payment"] = (
        df["PaymentMethod"].isin([
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
    ).astype(int)

    # Block 5: Risk score feature 
    # A hand-crafted risk signal combining multiple weak signals
    df["churn_risk_score"] = (
        df["is_month_to_month"] * 3 +
        df["is_new_customer"] * 2 +
        (1 - df["has_security_services"]) * 1 +
        (1 - df["is_auto_payment"]) * 1 +
        df["is_high_value"] * 1
    )

    # Block 6: Internet service features 
    df["has_internet"] = (
        df["InternetService"] != "No"
    ).astype(int)
    df["has_fiber"] = (
        df["InternetService"] == "Fiber optic"
    ).astype(int)

    # Block 7: Encode remaining categorical columns 
    # ── Block 7: Encode remaining categorical columns ─────────────────────────
    categorical_cols = [
        "InternetService", "Contract", "PaymentMethod", "gender"
    ]
    # Save Churn before encoding so it doesn't get lost
    churn_col = df["Churn"].copy() if "Churn" in df.columns else None
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # Restore Churn after encoding
    if churn_col is not None:
        df["Churn"] = churn_col

    # Save to processed 
    config = load_config()
    processed_path = Path(__file__).resolve().parents[2] / config["data"]["processed_path"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    print(f"  Saved to {processed_path}")

    return df


# Run directly
if __name__ == "__main__":
    df = build_features()
    print("\nSample of new features:")
    new_cols = [
        "tenure_years", "is_new_customer", "total_services",
        "churn_risk_score", "charge_increase_ratio", "Churn"
    ]
    print(df[new_cols].head(10))