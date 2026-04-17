import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.data.loader import load_config, load_processed


#Paths
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


# Build risk segments
def build_risk_segments(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Clusters at-risk customers into behavioral segments
    using their SHAP values — not raw features.

    This means segments are defined by WHY customers churn,
    not just what they look like demographically.

    Returns dataframe with segment labels and descriptions.
    """
    config = load_config()

    if df is None:
        df = load_processed()

    model         = joblib.load(MODEL_DIR / "best_model.joblib")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")

    # Get at-risk customers only
    X = df[[c for c in feature_names if c in df.columns]]
    probs = model.predict_proba(X)[:, 1]

    # Focus on high-risk customers (churn prob > 0.45)
    at_risk_mask = probs >= 0.45
    X_risk       = X[at_risk_mask].copy()
    probs_risk   = probs[at_risk_mask]

    print(f"At-risk customers (prob >= 0.45): {len(X_risk)} of {len(X)}")

    # Compute SHAP values for at-risk customers 
    print("Computing SHAP values for segmentation...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_risk)
    shap_df     = pd.DataFrame(shap_values, columns=feature_names)

    # Cluster on SHAP values 
    n_segments = config["shap"]["n_segments"]
    scaler     = StandardScaler()
    shap_scaled = scaler.fit_transform(shap_df)

    kmeans = KMeans(
        n_clusters  = n_segments,
        random_state = config["data"]["random_state"],
        n_init      = 10
    )
    cluster_labels = kmeans.fit_predict(shap_scaled)

    # Build result dataframe 
    result = X_risk.copy()
    result["churn_prob"]   = probs_risk
    result["segment_id"]   = cluster_labels
    result["segment_name"] = "unknown"

    # Name each segment by its dominant SHAP features 
    segment_profiles = {}

    for seg_id in range(n_segments):
        mask         = cluster_labels == seg_id
        seg_shap     = shap_df[mask]
        seg_features = result[mask]

        # Top 3 SHAP drivers for this segment
        mean_shap    = seg_shap.mean()
        top_features = mean_shap.abs().sort_values(ascending=False).head(3)
        top_names    = top_features.index.tolist()

        # Name the segment based on dominant features
        name = _name_segment(top_names, seg_features)
        result.loc[result["segment_id"] == seg_id, "segment_name"] = name

        segment_profiles[seg_id] = {
            "name":          name,
            "size":          int(mask.sum()),
            "avg_churn_prob": round(float(probs_risk[mask].mean()), 3),
            "top_drivers":   top_names
        }

    # Print segment summary
    print(f"\nCustomer Risk Segments ({n_segments} segments):")
    print("-" * 55)
    for seg_id, profile in segment_profiles.items():
        print(f"  Segment {seg_id}: {profile['name']}")
        print(f"    Size:          {profile['size']} customers")
        print(f"    Avg churn prob: {profile['avg_churn_prob']:.1%}")
        print(f"    Top drivers:   {', '.join(profile['top_drivers'][:2])}")
        print()

    # Save segments 
    result.to_parquet(MODEL_DIR / "risk_segments.parquet", index=False)
    print(f"Saved segments to models/risk_segments.parquet")

    return result, segment_profiles


# Name a segment based on its top SHAP drivers 
def _name_segment(top_features: list, seg_df: pd.DataFrame) -> str:

    feature_str = " ".join(top_features).lower()

    if "is_month_to_month" in feature_str and "is_new_customer" in feature_str:
        return "new-and-uncommitted"
    elif "is_month_to_month" in feature_str and "monthly_charges" in feature_str:
        return "price-sensitive"
    elif "has_fiber" in feature_str or "has_internet" in feature_str:
        return "service-dissatisfied"
    elif "tenure" in feature_str and "is_long_term" not in feature_str:
        return "early-lifecycle"
    elif "churn_risk_score" in feature_str:
        return "multi-risk-factor"
    elif "total_services" in feature_str or "has_security" in feature_str:
        return "low-engagement"
    else:
        return f"segment-{top_features[0][:12]}"


# Run directly 
if __name__ == "__main__":
    result, profiles = build_risk_segments()
    print(f"\nTotal at-risk customers segmented: {len(result)}")
    print(f"Segment distribution:")
    print(result["segment_name"].value_counts().to_string())