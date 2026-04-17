import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from src.data.loader import load_config, load_processed


# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


# ── Survival Model Class ───────────────────────────────────────────────────────
class SurvivalModel:
    """
    Predicts not just IF a customer will churn but WHEN.
    Uses Cox Proportional Hazards model from the lifelines library.

    Output: survival probability at 30, 60, 90 days per customer.
    Example: {"day_30": 0.82, "day_60": 0.61, "day_90": 0.44}
    Means: 82% chance of staying past 30 days, 44% chance past 90 days.
    """

    def __init__(self):
        self.cox_model   = CoxPHFitter(penalizer=0.1)
        self.km_model    = KaplanMeierFitter()
        self.config      = load_config()
        self.is_fitted   = False
        self.feature_cols = None

    # ── Fit the model ──────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame = None) -> "SurvivalModel":

        if df is None:
            df = load_processed()

        print("Fitting survival model...")

        time_col  = self.config["survival"]["time_column"]
        event_col = self.config["survival"]["event_column"]

        # ── Fit Kaplan-Meier (overall survival curve) ──────────────────────
        self.km_model.fit(
            durations  = df[time_col],
            event_observed = df[event_col],
            label      = "Overall survival"
        )

        # ── Select features for Cox model ──────────────────────────────────
        # Use only numeric columns, exclude leaky ones
        exclude_cols = [time_col, event_col]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        # Build survival dataframe: time + event + features
        survival_df = df[[time_col, event_col] + self.feature_cols].copy()

        # ── Fit Cox PH model ───────────────────────────────────────────────
        self.cox_model.fit(
            survival_df,
            duration_col = time_col,
            event_col    = event_col,
            show_progress = False
        )

        self.is_fitted = True

        print(f"Survival model fitted on {len(df)} customers")
        print(f"  Features used: {len(self.feature_cols)}")
        print(f"  Median survival time: {self.km_model.median_survival_time_:.1f} months")

        # Save model
        self._save()

        return self

    # ── Predict survival probabilities at specific time points ─────────────────
    def predict_survival_at_days(
        self,
        customer: pd.DataFrame,
        days: list = None
    ) -> dict:

        if not self.is_fitted:
            self._load()

        if days is None:
            days = self.config["survival"]["predict_at_days"]

        # Align customer features to match training
        for col in self.feature_cols:
            if col not in customer.columns:
                customer[col] = 0
        customer_features = customer[self.feature_cols]

        # Get survival function for this customer
        survival_func = self.cox_model.predict_survival_function(
            customer_features
        )

        result = {}
        for day in days:
            # Find closest time index
            times = survival_func.index.values
            closest_time = times[np.argmin(np.abs(times - day))]
            prob = float(survival_func.loc[closest_time].values[0])
            result[f"survival_day_{day}"] = round(prob, 4)

        # Also compute churn probability (1 - survival)
        result["churn_prob_day_30"] = round(1 - result["survival_day_30"], 4)
        result["churn_prob_day_60"] = round(1 - result["survival_day_60"], 4)
        result["churn_prob_day_90"] = round(1 - result["survival_day_90"], 4)

        return result

    # ── Get top risk factors from Cox model ────────────────────────────────────
    def get_risk_factors(self, top_n: int = 10) -> pd.DataFrame:
        if not self.is_fitted:
            self._load()

        summary = self.cox_model.summary.copy()
        summary = summary[["coef", "exp(coef)", "p"]].copy()
        summary.columns = ["coefficient", "hazard_ratio", "p_value"]
        summary = summary.sort_values("hazard_ratio", ascending=False)

        print("\nTop churn risk factors (hazard ratio > 1 = increases churn risk):")
        print(summary.head(top_n).to_string())

        return summary

    # ── Save and load ──────────────────────────────────────────────────────────
    def _save(self):
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self, MODEL_DIR / "survival_model.joblib")
        print(f"  Saved to models/survival_model.joblib")

    def _load(self):
        model_path = MODEL_DIR / "survival_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                "Survival model not found. Run: python -m src.models.survival"
            )
        loaded = joblib.load(model_path)
        self.cox_model    = loaded.cox_model
        self.km_model     = loaded.km_model
        self.feature_cols = loaded.feature_cols
        self.is_fitted    = True


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.features import build_features

    # Load and prepare data
    df = load_processed()

    # Fit survival model
    model = SurvivalModel()
    model.fit(df)

    # Show risk factors
    model.get_risk_factors(top_n=10)

    # Test prediction on a high risk customer
    print("\nSurvival prediction for high-risk customer:")
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
    }])

    # Fill missing feature columns with 0
    result = model.predict_survival_at_days(test)
    print(f"  Survival at 30 days:  {result['survival_day_30']:.1%}")
    print(f"  Survival at 60 days:  {result['survival_day_60']:.1%}")
    print(f"  Survival at 90 days:  {result['survival_day_90']:.1%}")
    print(f"  Churn prob by day 30: {result['churn_prob_day_30']:.1%}")
    print(f"  Churn prob by day 60: {result['churn_prob_day_60']:.1%}")