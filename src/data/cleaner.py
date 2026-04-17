import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import load_config, load_raw


# ── Main clean function ────────────────────────────────────────────────────────
def clean(df: pd.DataFrame = None, save: bool = None) -> pd.DataFrame:

    is_single_prediction = df is not None and len(df) < 10

    if df is None:
        df = load_raw()
        save = True
    elif save is None:
        save = False

    if not is_single_prediction:
        print("Starting data cleaning...")

    original_shape = df.shape

    # ── Step 1: Fix column names ───────────────────────────────────────────────
    df.columns = df.columns.str.strip()

    # ── Step 2: Fix TotalCharges ──────────────────────────────────────────────
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # ── Step 3: Fix target column (only if present) ───────────────────────────
    if "Churn" in df.columns:
        if df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # ── Step 4: Fix SeniorCitizen ─────────────────────────────────────────────
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # ── Step 5: Strip whitespace from all string columns ─────────────────────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # ── Step 6: Standardize binary Yes/No columns ────────────────────────────
    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines"
    ]
    for col in binary_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0})

    # ── Step 7: Drop customerID if present ───────────────────────────────────
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # ── Step 8: Remove duplicates (only for full dataset) ────────────────────
    if not is_single_prediction:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            print(f"  Removed {before - after} duplicate rows")

    # ── Step 9: Reset index ───────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    # ── Print summary (only for full dataset) ────────────────────────────────
    if not is_single_prediction:
        print(f"Cleaning complete: {original_shape} → {df.shape}")
        print(f"  Null values remaining: {df.isnull().sum().sum()}")
        if "Churn" in df.columns:
            print(f"  Churn rate: {df['Churn'].mean():.1%}")

    # ── Save to interim (only for full dataset) ───────────────────────────────
    if save:
        config = load_config()
        interim_path = Path(__file__).resolve().parents[2] / config["data"]["interim_path"]
        interim_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(interim_path, index=False)
        print(f"  Saved to {interim_path}")

    return df


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clean()