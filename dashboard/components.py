import plotly.graph_objects as go
import streamlit as st
import pandas as pd


def risk_badge(risk_tier: str) -> str:
    colors = {
        "critical": "🔴",
        "high":     "🟠",
        "medium":   "🟡",
        "low":      "🟢"
    }
    return f"{colors.get(risk_tier, '⚪')} {risk_tier.upper()}"


def shap_bar_chart(factors: list) -> go.Figure:
    if not factors:
        return None

    df = pd.DataFrame(factors)
    colors = [
        "#E24B4A" if d == "increases churn risk" else "#1D9E75"
        for d in df["direction"]
    ]

    fig = go.Figure(go.Bar(
        x            = df["magnitude"],
        y            = df["feature"],
        orientation  = "h",
        marker_color = colors
    ))
    fig.update_layout(
        height      = 300,
        margin      = dict(t=10, b=10, l=10, r=10),
        xaxis_title = "SHAP Magnitude",
        yaxis_title = "",
        showlegend  = False
    )
    return fig


def survival_curve_plot(survival_data: dict) -> go.Figure:
    days  = [30, 60, 90]
    probs = [
        survival_data.get("survival_day_30", 0),
        survival_data.get("survival_day_60", 0),
        survival_data.get("survival_day_90", 0),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = days,
        y    = probs,
        mode = "lines+markers",
        name = "Survival probability",
        line = dict(color="#378ADD", width=2),
        marker = dict(size=8)
    ))
    fig.update_layout(
        height      = 250,
        margin      = dict(t=10, b=10, l=10, r=10),
        xaxis_title = "Days",
        yaxis_title = "Survival probability",
        yaxis       = dict(range=[0, 1]),
        showlegend  = False
    )
    return fig


def retention_action_card(action: dict, index: int):
    type_colors = {
        "discount": "🎯",
        "loyalty":  "⭐",
        "support":  "🛠️",
        "outreach": "📞",
        "upgrade":  "⬆️"
    }
    icon = type_colors.get(action.get("type", ""), "💡")
    with st.expander(f"{icon} Action {index}: {action.get('title', '')}"):
        st.markdown(f"**Type:** {action.get('type', '')}")
        st.markdown(f"**Message:** {action.get('message', '')}")
        st.markdown(f"**Offer:** {action.get('offer', '')}")
        st.markdown(f"**Expected Impact:** {action.get('expected_impact', '')}")