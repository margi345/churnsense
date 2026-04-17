import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_processed
from src.models.predict import predict, predict_batch
from src.explainability.shap_explainer import explain_local
from src.retention.llm_engine import RetentionEngine
from src.data.cleaner import clean
from src.data.features import build_features

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ChurnSense",
    page_icon  = "📡",
    layout     = "wide"
)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = load_processed()
    return df

@st.cache_data
def get_predictions(df):
    MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
    model         = joblib.load(MODEL_DIR / "best_model.joblib")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    X = df[[c for c in feature_names if c in df.columns]]
    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["churn_score"] = probs.round(4)
    df["risk_tier"] = pd.cut(
        probs,
        bins=[-np.inf, 0.25, 0.45, 0.70, np.inf],
        labels=["low", "medium", "high", "critical"]
    )
    return df

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📡 ChurnSense — Retention Intelligence Dashboard")
st.markdown("AI-powered churn prediction, explanation, and retention planning")
st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading data and predictions..."):
    df      = load_data()
    df_pred = get_predictions(df)

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.title("Filters")
risk_filter = st.sidebar.multiselect(
    "Risk Tier",
    options = ["critical", "high", "medium", "low"],
    default = ["critical", "high"]
)

min_score = st.sidebar.slider(
    "Minimum Churn Score",
    min_value = 0.0,
    max_value = 1.0,
    value     = 0.45,
    step      = 0.05
)

# ── Apply filters ──────────────────────────────────────────────────────────────
filtered = df_pred[
    (df_pred["risk_tier"].isin(risk_filter)) &
    (df_pred["churn_score"] >= min_score)
].copy()

# ── Metric cards ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers",    f"{len(df_pred):,}")
col2.metric("At-Risk Customers",  f"{len(filtered):,}")
col3.metric("Avg Churn Score",    f"{filtered['churn_score'].mean():.1%}" if len(filtered) > 0 else "N/A")
col4.metric("Critical Risk",      f"{len(df_pred[df_pred['risk_tier']=='critical']):,}")

st.divider()

# ── Two column layout ──────────────────────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    st.subheader("At-Risk Customers")
    display_cols = ["tenure", "MonthlyCharges", "churn_score", "risk_tier"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols]
        .sort_values("churn_score", ascending=False)
        .head(50)
        .reset_index(drop=True),
        use_container_width = True,
        height = 400
    )

with right:
    st.subheader("Risk Distribution")
    risk_counts = df_pred["risk_tier"].value_counts()
    colors = {
        "critical": "#E24B4A",
        "high":     "#EF9F27",
        "medium":   "#378ADD",
        "low":      "#1D9E75"
    }
    fig = go.Figure(go.Pie(
        labels = risk_counts.index,
        values = risk_counts.values,
        marker_colors = [colors.get(k, "#888") for k in risk_counts.index],
        hole   = 0.4
    ))
    fig.update_layout(
        margin = dict(t=0, b=0, l=0, r=0),
        height = 300,
        showlegend = True
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Customer detail section ────────────────────────────────────────────────────
st.subheader("Customer Deep Dive")
st.markdown("Enter a customer index from the table above to see full analysis")

customer_idx = st.number_input(
    "Customer Index",
    min_value = 0,
    max_value = len(filtered) - 1 if len(filtered) > 0 else 0,
    value     = 0,
    step      = 1
)

if len(filtered) > 0 and st.button("Analyze Customer", type="primary"):
    customer_row = filtered.iloc[customer_idx]

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Score",  f"{customer_row['churn_score']:.1%}")
    c2.metric("Risk Tier",    str(customer_row['risk_tier']).upper())
    c3.metric("Tenure",       f"{customer_row.get('tenure', 'N/A')} months")

    # ── SHAP explanation ───────────────────────────────────────────────────
    st.markdown("#### Why is this customer at risk?")
    with st.spinner("Computing SHAP explanation..."):
        try:
            customer_df = pd.DataFrame([customer_row])
            factors     = explain_local(customer_df, top_n=5)

            factor_df = pd.DataFrame(factors)
            colors_shap = ["#E24B4A" if d == "increases churn risk"
                          else "#1D9E75" for d in factor_df["direction"]]

            fig2 = go.Figure(go.Bar(
                x           = factor_df["magnitude"],
                y           = factor_df["feature"],
                orientation = "h",
                marker_color = colors_shap
            ))
            fig2.update_layout(
                height       = 300,
                margin       = dict(t=10, b=10),
                xaxis_title  = "SHAP Magnitude",
                yaxis_title  = ""
            )
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"SHAP error: {e}")

    # ── LLM Retention plan ─────────────────────────────────────────────────
    st.markdown("#### AI Retention Plan")
    with st.spinner("Generating personalized retention strategy..."):
        try:
            engine  = RetentionEngine()
            profile = customer_row.to_dict()
            profile["churn_score"] = float(customer_row["churn_score"])
            profile["risk_tier"]   = str(customer_row["risk_tier"])

            factors_dict = explain_local(
                pd.DataFrame([customer_row]), top_n=5
            )
            plan = engine.generate(profile, factors_dict)

            st.info(f"**Segment:** {plan.get('segment')} | "
                   f"**Urgency:** {plan.get('urgency').upper()} | "
                   f"{plan.get('summary')}")

            for i, action in enumerate(plan.get("actions", []), 1):
                with st.expander(f"Action {i}: {action.get('title')}"):
                    st.markdown(f"**Type:** {action.get('type')}")
                    st.markdown(f"**Message:** {action.get('message')}")
                    st.markdown(f"**Offer:** {action.get('offer')}")
                    st.markdown(f"**Expected Impact:** {action.get('expected_impact')}")

        except Exception as e:
            st.error(f"Retention plan error: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "ChurnSense v1.0.0 — Built with XGBoost, SHAP, Lifelines, and GPT-4o-mini"
)