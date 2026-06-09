"""Plotly and Streamlit presentation helpers for the premium dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


COLOR_BG = "#07111f"
COLOR_PANEL = "rgba(16, 28, 48, 0.72)"
COLOR_BORDER = "rgba(148, 163, 184, 0.22)"
COLOR_TEXT = "#e5eefb"
COLOR_MUTED = "#94a3b8"
COLOR_ACCENT = "#38bdf8"
COLOR_GOOD = "#22c55e"
COLOR_WARN = "#f59e0b"
COLOR_BAD = "#ef4444"


def inject_premium_css() -> None:
    """Apply dark premium banking styling."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(56,189,248,.20), transparent 28rem),
                linear-gradient(135deg, #06101d 0%, #0b1728 48%, #0f172a 100%);
            color: {COLOR_TEXT};
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(8,18,34,.96), rgba(15,23,42,.96));
            border-right: 1px solid {COLOR_BORDER};
        }}
        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1420px;
        }}
        h1, h2, h3 {{
            letter-spacing: 0;
            color: #f8fbff;
        }}
        .hero {{
            padding: 1.25rem 1.4rem;
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(15,23,42,.82), rgba(15,32,57,.74));
            box-shadow: 0 24px 70px rgba(0,0,0,.28);
            margin-bottom: 1rem;
        }}
        .hero-title {{
            font-size: 2rem;
            font-weight: 760;
            margin-bottom: .25rem;
        }}
        .hero-subtitle {{
            color: {COLOR_MUTED};
            font-size: .98rem;
        }}
        .metric-card {{
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 1rem;
            background: {COLOR_PANEL};
            box-shadow: 0 18px 50px rgba(0,0,0,.22);
            min-height: 118px;
        }}
        .metric-label {{
            color: {COLOR_MUTED};
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .04em;
        }}
        .metric-value {{
            color: #ffffff;
            font-size: 1.62rem;
            font-weight: 760;
            margin-top: .4rem;
        }}
        .metric-help {{
            color: {COLOR_ACCENT};
            font-size: .82rem;
            margin-top: .35rem;
        }}
        .risk-low {{ color: {COLOR_GOOD}; }}
        .risk-medium {{ color: {COLOR_WARN}; }}
        .risk-high {{ color: {COLOR_BAD}; }}
        div[data-testid="stMetric"] {{
            background: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: .8rem 1rem;
        }}
        .stButton > button {{
            width: 100%;
            border-radius: 8px;
            background: linear-gradient(135deg, #0ea5e9, #2563eb);
            color: white;
            border: 0;
            font-weight: 700;
            padding: .72rem 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='hero'><div class='hero-title'>{title}</div><div class='hero-subtitle'>{subtitle}</div></div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-help'>{help_text}</div></div>",
        unsafe_allow_html=True,
    )


def risk_color(category: str) -> str:
    if category == "Low Risk":
        return COLOR_GOOD
    if category == "Medium Risk":
        return COLOR_WARN
    return COLOR_BAD


def probability_gauge(probability: float, title: str = "Default Probability") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"color": COLOR_TEXT}},
            title={"text": title, "font": {"color": COLOR_TEXT}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLOR_MUTED},
                "bar": {"color": risk_color("Low Risk" if probability < 0.2 else "Medium Risk" if probability < 0.5 else "High Risk")},
                "bgcolor": "rgba(15,23,42,.9)",
                "borderwidth": 1,
                "bordercolor": COLOR_BORDER,
                "steps": [
                    {"range": [0, 20], "color": "rgba(34,197,94,.28)"},
                    {"range": [20, 50], "color": "rgba(245,158,11,.28)"},
                    {"range": [50, 100], "color": "rgba(239,68,68,.28)"},
                ],
            },
        )
    )
    return apply_plot_theme(fig, height=290)


def apply_plot_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,.35)",
        font={"color": COLOR_TEXT},
        margin={"l": 24, "r": 24, "t": 54, "b": 32},
        legend={"orientation": "h", "y": -0.18},
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.14)", zerolinecolor="rgba(148,163,184,.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.14)", zerolinecolor="rgba(148,163,184,.18)")
    return fig


def roc_curve_figure(metadata: dict[str, Any]) -> go.Figure:
    roc_payload = metadata.get("test_roc_curve", {})
    fig = go.Figure()
    if roc_payload:
        fig.add_trace(go.Scatter(x=roc_payload["fpr"], y=roc_payload["tpr"], mode="lines", name="Best model", line={"color": COLOR_ACCENT, "width": 3}))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"color": COLOR_MUTED, "dash": "dash"}))
    fig.update_layout(title="Interactive ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return apply_plot_theme(fig)


def confusion_matrix_figure(metadata: dict[str, Any]) -> go.Figure:
    matrix = metadata.get("test_confusion_matrix", [[0, 0], [0, 0]])
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", labels={"x": "Predicted", "y": "Actual", "color": "Count"})
    fig.update_layout(title="Confusion Matrix")
    return apply_plot_theme(fig, height=360)


def feature_importance_figure(path: str | Path) -> go.Figure:
    importance_path = Path(path)
    if importance_path.exists():
        data = pd.read_csv(importance_path).head(20).sort_values("importance")
    else:
        data = pd.DataFrame({"feature": [], "importance": []})
    fig = px.bar(data, x="importance", y="feature", orientation="h", title="Interactive Feature Importance", color="importance", color_continuous_scale="Blues")
    return apply_plot_theme(fig, height=540)


def risk_distribution_figure(probabilities: np.ndarray) -> go.Figure:
    data = pd.DataFrame({"default_probability": probabilities})
    fig = px.histogram(data, x="default_probability", nbins=40, title="Risk Distribution", color_discrete_sequence=[COLOR_ACCENT])
    fig.update_layout(xaxis_title="Default probability", yaxis_title="Customers")
    return apply_plot_theme(fig)


def income_distribution_figure(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="AMT_INCOME_TOTAL", nbins=45, title="Income Distribution", color_discrete_sequence=["#60a5fa"])
    fig.update_layout(xaxis_title="Income", yaxis_title="Customers")
    return apply_plot_theme(fig)


def segmentation_figure(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["segment"] = pd.cut(
        data["CREDIT_INCOME_RATIO"],
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["Conservative", "Balanced", "Leveraged", "Stretched"],
    )
    segment = data.groupby("segment", observed=False)["TARGET"].agg(default_rate="mean", customers="count").reset_index()
    fig = px.bar(segment, x="segment", y="default_rate", size="customers", title="Customer Segmentation by Credit Burden", color="default_rate", color_continuous_scale="Reds")
    fig.update_layout(xaxis_title="Segment", yaxis_title="Default rate")
    return apply_plot_theme(fig)


def correlation_matrix_figure(df: pd.DataFrame) -> go.Figure:
    cols = [
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
    ]
    corr = df[[col for col in cols if col in df.columns]].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Matrix")
    return apply_plot_theme(fig, height=520)
