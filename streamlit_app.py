from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_utils import (
    COLOR_BAD,
    COLOR_GOOD,
    COLOR_WARN,
    apply_plot_theme,
    confusion_matrix_figure,
    correlation_matrix_figure,
    feature_importance_figure,
    income_distribution_figure,
    inject_premium_css,
    probability_gauge,
    risk_distribution_figure,
    roc_curve_figure,
)
from model_utils import load_artifacts, predict_risk, shap_contributions
from utils import (
    TARGET,
    add_engineered_features,
    align_columns,
    integer_days_from_years,
    load_application_data,
    preprocess_input,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "application_train.csv"
MODEL_PATH = ROOT / "outputs" / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "outputs" / "models" / "model_metadata.json"
FEATURE_IMPORTANCE_PATH = ROOT / "outputs" / "figures" / "feature_importance_top_20.csv"

APP_TITLE = "Credit Risk Intelligence Platform"


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_premium_css()


# ============================================================
# PREMIUM CSS
# ============================================================

def inject_luxury_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --lux-bg-1: #050816;
            --lux-bg-2: #0B1020;
            --lux-card: rgba(255, 255, 255, 0.065);
            --lux-border: rgba(255, 255, 255, 0.145);
            --lux-gold: #D7B56D;
            --lux-gold-2: #F2DA9A;
            --lux-text: #F7F8FC;
            --lux-muted: #AAB2C5;
            --lux-red: #FF6B7A;
            --lux-green: #5CD6A2;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(215,181,109,0.18) 0, transparent 34%),
                radial-gradient(circle at 85% 18%, rgba(110,168,254,0.14) 0, transparent 30%),
                linear-gradient(135deg, var(--lux-bg-1) 0%, var(--lux-bg-2) 54%, #070A12 100%);
            color: var(--lux-text);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.025)),
                #060915;
            border-right: 1px solid var(--lux-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--lux-text);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 8px 12px;
            margin: 6px 0;
            transition: all 0.2s ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(215,181,109,0.13);
            border-color: rgba(215,181,109,0.35);
            transform: translateX(2px);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        .lux-hero {
            position: relative;
            padding: 30px 32px;
            margin-bottom: 22px;
            border-radius: 30px;
            border: 1px solid rgba(215,181,109,0.24);
            background:
                linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.035)),
                radial-gradient(circle at 85% 20%, rgba(215,181,109,0.26), transparent 28%);
            box-shadow: 0 26px 70px rgba(0,0,0,0.36);
            overflow: hidden;
        }

        .lux-eyebrow {
            color: var(--lux-gold-2);
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-size: 0.74rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .lux-title {
            color: var(--lux-text);
            font-size: clamp(2.1rem, 4vw, 4.2rem);
            line-height: 1.02;
            font-weight: 900;
            letter-spacing: -0.06em;
            margin: 0;
        }

        .lux-subtitle {
            color: var(--lux-muted);
            max-width: 880px;
            margin-top: 14px;
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .lux-card {
            padding: 22px 22px;
            border-radius: 24px;
            border: 1px solid var(--lux-border);
            background: linear-gradient(145deg, var(--lux-card), rgba(255,255,255,0.025));
            box-shadow: 0 22px 60px rgba(0,0,0,0.28);
            height: 100%;
        }

        .lux-card h4 {
            margin: 0 0 8px 0;
            color: var(--lux-text);
            font-size: 0.94rem;
            letter-spacing: 0.02em;
        }

        .lux-card p {
            color: var(--lux-muted);
            line-height: 1.55;
            margin: 0;
            font-size: 0.92rem;
        }

        .lux-metric {
            padding: 18px 18px;
            border-radius: 22px;
            border: 1px solid rgba(255,255,255,0.12);
            background:
                linear-gradient(145deg, rgba(255,255,255,0.105), rgba(255,255,255,0.035)),
                radial-gradient(circle at top right, rgba(215,181,109,0.16), transparent 38%);
            box-shadow: 0 16px 44px rgba(0,0,0,0.28);
            min-height: 128px;
        }

        .lux-metric-label {
            color: var(--lux-muted);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }

        .lux-metric-value {
            color: var(--lux-text);
            font-size: 1.72rem;
            line-height: 1.15;
            font-weight: 900;
            letter-spacing: -0.04em;
            margin-top: 10px;
        }

        .lux-metric-note {
            color: var(--lux-gold-2);
            font-size: 0.78rem;
            margin-top: 8px;
        }

        .section-heading {
            margin: 22px 0 12px 0;
            color: var(--lux-text);
            font-weight: 850;
            letter-spacing: -0.03em;
            font-size: 1.35rem;
        }

        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.12);
        }

        .stButton > button, div[data-testid="stFormSubmitButton"] button {
            border-radius: 999px;
            border: 1px solid rgba(215,181,109,0.45);
            background: linear-gradient(135deg, #C69A3B, #F2DA9A);
            color: #07101E;
            font-weight: 850;
            padding: 0.68rem 1.35rem;
            box-shadow: 0 14px 34px rgba(215,181,109,0.22);
        }

        .stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
            border-color: rgba(242,218,154,0.8);
            box-shadow: 0 18px 44px rgba(215,181,109,0.32);
            transform: translateY(-1px);
        }

        .stAlert {
            border-radius: 18px;
        }

        hr {
            border-color: rgba(255,255,255,0.1);
            margin: 1.3rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_luxury_css()


# ============================================================
# STREAMLIT VERSION-SAFE WRAPPERS
# ============================================================

def plot_chart(fig: go.Figure, key: str | None = None) -> None:
    kwargs = {}
    if key is not None:
        kwargs["key"] = key

    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def show_df(df: pd.DataFrame, key: str | None = None, height: int | None = None) -> None:
    """
    Important fix:
    Do NOT pass height=None to Streamlit.
    New Streamlit versions reject None.
    """
    kwargs = {}

    if key is not None:
        kwargs["key"] = key

    if height is not None:
        kwargs["height"] = height

    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def show_image(path: str) -> None:
    try:
        st.image(path, width="stretch")
    except TypeError:
        st.image(path, use_container_width=True)


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def cached_artifacts():
    return load_artifacts(MODEL_PATH, METADATA_PATH)


@st.cache_data(show_spinner=False)
def cached_data(nrows: int = 60000) -> pd.DataFrame:
    data = load_application_data(DATA_PATH, nrows=nrows)

    if "DAYS_EMPLOYED" in data.columns:
        data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(365243, np.nan)

    return add_engineered_features(data)


@st.cache_data(show_spinner=False)
def cached_reference() -> pd.DataFrame:
    data = load_application_data(DATA_PATH, nrows=1500)

    if TARGET in data.columns:
        data = data.drop(columns=[TARGET])

    if "DAYS_EMPLOYED" in data.columns:
        data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(365243, np.nan)

    return data


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_series(data: pd.DataFrame, column: str, fallback: float = 0.0) -> pd.Series:
    if column not in data.columns:
        return pd.Series([fallback] * len(data), index=data.index)
    return pd.to_numeric(data[column], errors="coerce")


def safe_mean(data: pd.DataFrame, column: str, fallback: float = 0.0) -> float:
    if column not in data.columns:
        return fallback

    value = pd.to_numeric(data[column], errors="coerce").mean()
    return fallback if pd.isna(value) else float(value)


def safe_rate(data: pd.DataFrame, column: str, fallback: float = 0.0) -> float:
    if column not in data.columns:
        return fallback

    value = pd.to_numeric(data[column], errors="coerce").mean()
    return fallback if pd.isna(value) else float(value)


def safe_default(reference: pd.DataFrame, column: str, fallback: float | str) -> Any:
    if column not in reference.columns:
        return fallback

    clean = reference[column].dropna()

    if clean.empty:
        return fallback

    value = clean.iloc[0]
    return fallback if pd.isna(value) else value


def safe_metric(metrics: list[dict], names: list[str], fallback: float = 0.0) -> float:
    if not metrics:
        return fallback

    first = metrics[0]

    for name in names:
        if name in first and pd.notna(first[name]):
            try:
                return float(first[name])
            except (TypeError, ValueError):
                return fallback

    return fallback


def available_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in data.columns]


def get_feature_names(metadata: dict) -> list[str]:
    features = metadata.get("features", [])

    if not isinstance(features, list) or not features:
        raise ValueError("No model feature list found in model_metadata.json.")

    return features


def risk_colour(category: str) -> str:
    return {
        "Low Risk": COLOR_GOOD,
        "Medium Risk": COLOR_WARN,
        "High Risk": COLOR_BAD,
    }.get(category, "#D7B56D")


def money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}K"

    return f"{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.2%}"


def luxury_hero(title: str, subtitle: str, eyebrow: str = "Premium Banking AI") -> None:
    st.markdown(
        f"""
        <div class="lux-hero">
            <div class="lux-eyebrow">{eyebrow}</div>
            <h1 class="lux-title">{title}</h1>
            <div class="lux-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def luxury_metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="lux-metric">
            <div class="lux-metric-label">{label}</div>
            <div class="lux-metric-value">{value}</div>
            <div class="lux-metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def luxury_info_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="lux-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PREMIUM FIGURES
# ============================================================

def premium_segmentation_figure(data: pd.DataFrame) -> go.Figure:
    """
    Fixed replacement for old segmentation_figure.

    Old problem:
    px.bar(..., size="customers") causes error.

    Fixed:
    px.scatter supports size="customers".
    """
    working = data.copy()

    if "CREDIT_INCOME_RATIO" not in working.columns:
        income = safe_series(working, "AMT_INCOME_TOTAL").replace(0, np.nan)
        credit = safe_series(working, "AMT_CREDIT")
        working["CREDIT_INCOME_RATIO"] = credit / income

    working["segment"] = pd.cut(
        pd.to_numeric(working["CREDIT_INCOME_RATIO"], errors="coerce"),
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["Prime", "Balanced", "Leveraged", "Critical"],
    )

    if TARGET not in working.columns:
        working[TARGET] = np.nan

    if "AMT_CREDIT" not in working.columns:
        working["AMT_CREDIT"] = np.nan

    segment = (
        working.dropna(subset=["segment"])
        .groupby("segment", observed=False)
        .agg(
            customers=("segment", "size"),
            default_rate=(TARGET, "mean"),
            avg_credit=("AMT_CREDIT", "mean"),
        )
        .reset_index()
    )

    if segment.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No segmentation data available.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        return apply_plot_theme(fig)

    fig = px.scatter(
        segment,
        x="segment",
        y="default_rate",
        size="customers",
        color="default_rate",
        hover_data={
            "customers": ":,",
            "avg_credit": ":,.0f",
            "default_rate": ":.2%",
            "segment": False,
        },
        title="Customer Segmentation by Credit Burden",
        color_continuous_scale=["#5CD6A2", "#D7B56D", "#FF6B7A"],
        size_max=58,
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=1.5, color="rgba(255,255,255,0.72)"),
            opacity=0.92,
        )
    )

    fig.update_layout(
        yaxis_tickformat=".1%",
        xaxis_title="Risk Segment",
        yaxis_title="Observed Default Rate",
        margin=dict(l=20, r=20, t=70, b=35),
    )

    return apply_plot_theme(fig)


def portfolio_quality_figure(data: pd.DataFrame) -> go.Figure:
    income = safe_series(data, "AMT_INCOME_TOTAL").replace(0, np.nan)
    credit = safe_series(data, "AMT_CREDIT")
    ratio = pd.to_numeric(data.get("CREDIT_INCOME_RATIO", credit / income), errors="coerce")

    profile = pd.DataFrame(
        {
            "Band": ["Prime", "Balanced", "Leveraged", "Critical"],
            "Customers": [
                int((ratio <= 2).sum()),
                int(((ratio > 2) & (ratio <= 4)).sum()),
                int(((ratio > 4) & (ratio <= 6)).sum()),
                int((ratio > 6).sum()),
            ],
        }
    )

    fig = px.pie(
        profile,
        values="Customers",
        names="Band",
        hole=0.62,
        title="Portfolio Quality Mix",
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")

    fig.update_layout(
        margin=dict(l=20, r=20, t=70, b=20),
        showlegend=True,
    )

    return apply_plot_theme(fig)


def affordability_scatter_figure(data: pd.DataFrame, sample_size: int = 5000) -> go.Figure:
    required = ["AMT_INCOME_TOTAL", "AMT_CREDIT"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required columns: {', '.join(missing)}",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return apply_plot_theme(fig)

    plot_data = (
        data.sample(min(sample_size, len(data)), random_state=42)
        if len(data) > sample_size
        else data.copy()
    )

    color_col = TARGET if TARGET in plot_data.columns else None

    fig = px.scatter(
        plot_data,
        x="AMT_INCOME_TOTAL",
        y="AMT_CREDIT",
        color=color_col,
        opacity=0.52,
        title="Income vs Credit Exposure",
        labels={
            "AMT_INCOME_TOTAL": "Applicant Income",
            "AMT_CREDIT": "Credit Exposure",
            TARGET: "Default",
        },
    )

    fig.update_layout(
        xaxis_tickformat=",",
        yaxis_tickformat=",",
        margin=dict(l=20, r=20, t=70, b=35),
    )

    return apply_plot_theme(fig)


def predicted_risk_table(
    model,
    data: pd.DataFrame,
    metadata: dict,
    rows: int = 12,
) -> pd.DataFrame:
    feature_names = get_feature_names(metadata)

    sample = data.drop(columns=[TARGET], errors="ignore").head(2500)
    aligned = align_columns(sample, feature_names)

    probabilities = model.predict_proba(aligned)[:, 1]

    table = sample.copy()
    table["Predicted Default Probability"] = probabilities

    table["Risk Tier"] = pd.cut(
        table["Predicted Default Probability"],
        bins=[-0.01, 0.20, 0.50, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    preferred = [
        "Risk Tier",
        "Predicted Default Probability",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
    ]

    cols = available_columns(table, preferred)

    result = (
        table[cols]
        .sort_values("Predicted Default Probability", ascending=False)
        .head(rows)
    )

    if "Predicted Default Probability" in result.columns:
        result["Predicted Default Probability"] = result[
            "Predicted Default Probability"
        ].map(lambda x: f"{x:.2%}")

    return result


# ============================================================
# SIDEBAR
# ============================================================

def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 18px 14px 14px; border-radius: 24px;
                        background: linear-gradient(145deg, rgba(215,181,109,0.18), rgba(255,255,255,0.04));
                        border: 1px solid rgba(215,181,109,0.24); margin-bottom: 18px;">
                <div style="font-size: 1.8rem;">🏦</div>
                <div style="font-weight: 900; font-size: 1.05rem; margin-top: 8px;">Credit Risk Intelligence</div>
                <div style="color: #AAB2C5; font-size: 0.82rem; margin-top: 4px;">Executive AI Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Customer Prediction",
                "Risk Analysis",
                "Explainable AI",
                "Model Performance",
                "Business Insights",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption(
            "Premium build: stable charts, executive UI, safer missing-column handling, and updated Streamlit width support."
        )

    return page


# ============================================================
# CUSTOMER INPUT FORM
# ============================================================

def select_from_reference(
    reference: pd.DataFrame,
    column: str,
    label: str,
) -> str | None:
    if column not in reference.columns:
        st.selectbox(label, ["Not available"], disabled=True)
        return None

    options = sorted(reference[column].dropna().astype(str).unique().tolist())

    if not options:
        st.selectbox(label, ["Not available"], disabled=True)
        return None

    return st.selectbox(label, options)


def build_customer_input(
    reference: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    if reference.empty:
        st.error("Reference data is empty. Check the source CSV file.")
        return pd.DataFrame()

    base = reference.iloc[[0]].copy()

    for column in feature_names:
        if column not in base.columns:
            base[column] = np.nan

    with st.form("risk_prediction_form"):
        section("Applicant Profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.slider("Age", 18, 75, 35)

            income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=float(safe_default(reference, "AMT_INCOME_TOTAL", 150000.0)),
                step=10000.0,
            )

            gender = select_from_reference(reference, "CODE_GENDER", "Gender")

        with c2:
            credit = st.number_input(
                "Credit Amount",
                min_value=0.0,
                value=float(safe_default(reference, "AMT_CREDIT", 500000.0)),
                step=10000.0,
            )

            annuity = st.number_input(
                "Annuity",
                min_value=0.0,
                value=float(safe_default(reference, "AMT_ANNUITY", 25000.0)),
                step=1000.0,
            )

            education = select_from_reference(
                reference,
                "NAME_EDUCATION_TYPE",
                "Education",
            )

        with c3:
            employment_years = st.slider("Employment Length", 0, 45, 5)

            family_status = select_from_reference(
                reference,
                "NAME_FAMILY_STATUS",
                "Family Status",
            )

            income_type = select_from_reference(
                reference,
                "NAME_INCOME_TYPE",
                "Income Type",
            )

        submitted = st.form_submit_button("Generate Executive Risk Decision")

    if not submitted:
        return pd.DataFrame()

    row = base.copy()
    index = row.index[0]

    row.loc[index, "DAYS_BIRTH"] = integer_days_from_years(age)
    row.loc[index, "DAYS_EMPLOYED"] = integer_days_from_years(employment_years)
    row.loc[index, "AMT_INCOME_TOTAL"] = float(income)
    row.loc[index, "AMT_CREDIT"] = float(credit)
    row.loc[index, "AMT_ANNUITY"] = float(annuity)

    if gender is not None:
        row.loc[index, "CODE_GENDER"] = gender

    if education is not None:
        row.loc[index, "NAME_EDUCATION_TYPE"] = education

    if family_status is not None:
        row.loc[index, "NAME_FAMILY_STATUS"] = family_status

    if income_type is not None:
        row.loc[index, "NAME_INCOME_TYPE"] = income_type

    try:
        return preprocess_input(row, feature_names)
    except Exception as exc:
        st.error(f"Input preprocessing failed: {exc}")
        return pd.DataFrame()


# ============================================================
# PAGES
# ============================================================

def show_dashboard(model, data: pd.DataFrame, metadata: dict) -> None:
    luxury_hero(
        "Executive Risk Dashboard",
        "A premium command center for portfolio health, underwriting exposure, default intelligence, and model confidence.",
        "Board-Level Portfolio Overview",
    )

    metrics = metadata.get("metrics", [])

    roc_auc = safe_metric(metrics, ["ROC AUC", "CV ROC AUC", "AUC"], 0.0)
    default_rate = safe_rate(data, TARGET, 0.0)
    avg_income = safe_mean(data, "AMT_INCOME_TOTAL", 0.0)
    avg_credit = safe_mean(data, "AMT_CREDIT", 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        luxury_metric("Customers Loaded", f"{len(data):,}", "active sample")

    with c2:
        luxury_metric("Avg Income", money(avg_income), "portfolio income")

    with c3:
        luxury_metric("Avg Credit", money(avg_credit), "credit exposure")

    with c4:
        luxury_metric("Default Rate", pct(default_rate), "observed target")

    with c5:
        luxury_metric(
            "Model ROC-AUC",
            f"{roc_auc:.3f}",
            metadata.get("best_model_name", "best model"),
        )

    st.markdown("")

    a, b, c = st.columns([1.15, 1, 1])

    with a:
        luxury_info_card(
            "Executive Summary",
            "This dashboard consolidates portfolio quality, exposure concentration, model strength, and customer risk tiers into one decision-ready view.",
        )

    with b:
        luxury_info_card(
            "Risk Governance",
            "Use model scores as a decision-support layer alongside affordability checks, policy rules, and manual review controls.",
        )

    with c:
        luxury_info_card(
            "Commercial Focus",
            "Prioritize low-risk growth, price medium-risk applications carefully, and route high-risk profiles to enhanced due diligence.",
        )

    section("Portfolio Intelligence")

    c1, c2 = st.columns(2)

    with c1:
        plot_chart(income_distribution_figure(data), key="dashboard_income_distribution")

    with c2:
        plot_chart(premium_segmentation_figure(data), key="dashboard_premium_segmentation")

    c3, c4 = st.columns(2)

    with c3:
        plot_chart(portfolio_quality_figure(data), key="dashboard_portfolio_quality")

    with c4:
        plot_chart(affordability_scatter_figure(data), key="dashboard_affordability")

    section("Highest Predicted-Risk Applications")

    try:
        show_df(
            predicted_risk_table(model, data, metadata),
            key="dashboard_risk_table",
            height=420,
        )
    except Exception as exc:
        st.info(f"Risk table could not be generated: {exc}")


def show_prediction(model, metadata: dict, reference: pd.DataFrame) -> None:
    luxury_hero(
        "Customer Prediction",
        "Generate a premium underwriting view with default probability, repayment probability, confidence, and aligned model features.",
        "Single Applicant Underwriting",
    )

    feature_names = get_feature_names(metadata)
    row = build_customer_input(reference, feature_names)

    if row.empty:
        st.info("Complete the applicant profile and press Generate Executive Risk Decision.")
        return

    try:
        _, probability, category, confidence = predict_risk(model, row)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    colour = risk_colour(category)

    c1, c2 = st.columns([1.15, 0.85])

    with c1:
        st.markdown(
            f"""
            <div class="lux-card">
                <div class="lux-eyebrow">Decision Signal</div>
                <h2 style="margin: 0; color: {colour}; font-size: 2.4rem;">{category}</h2>
                <p style="margin-top: 10px;">Risk category generated from the trained model using the aligned feature set.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        plot_chart(probability_gauge(probability), key="prediction_probability_gauge")

    with c2:
        luxury_metric("Prediction Confidence", pct(float(confidence)), "model certainty")
        st.markdown("")
        luxury_metric("Default Probability", pct(float(probability)), "risk of default")
        st.markdown("")
        luxury_metric("Repayment Probability", pct(1 - float(probability)), "expected good outcome")

    st.progress(min(max(float(probability), 0.0), 1.0))
    st.caption(
        "Decision rule: low risk below 20%, medium risk from 20% to 50%, high risk above 50%."
    )

    section("Aligned Prediction Features")

    display_cols = available_columns(
        row,
        [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
        ],
    )

    if display_cols:
        show_df(row[display_cols], key="prediction_features")
    else:
        st.info("No displayable aligned features found.")


def show_risk_analysis(model, data: pd.DataFrame, metadata: dict) -> None:
    luxury_hero(
        "Risk Analysis",
        "Segment portfolio exposure, identify affordability stress, and review risk-distribution patterns at scale.",
        "Portfolio Risk Analytics",
    )

    feature_names = get_feature_names(metadata)

    try:
        sample = data.drop(columns=[TARGET], errors="ignore").head(5000)
        aligned = align_columns(sample, feature_names)
        probabilities = model.predict_proba(aligned)[:, 1]
    except Exception as exc:
        st.error(f"Risk probabilities could not be generated: {exc}")
        return

    c1, c2 = st.columns(2)

    with c1:
        plot_chart(risk_distribution_figure(probabilities), key="risk_distribution")

    with c2:
        plot_chart(correlation_matrix_figure(data), key="correlation_matrix")

    section("Credit Exposure by Risk Segment")

    segmented = data.copy()

    if "CREDIT_INCOME_RATIO" not in segmented.columns:
        income = safe_series(segmented, "AMT_INCOME_TOTAL").replace(0, np.nan)
        segmented["CREDIT_INCOME_RATIO"] = safe_series(segmented, "AMT_CREDIT") / income

    segmented["Risk Segment"] = pd.cut(
        pd.to_numeric(segmented["CREDIT_INCOME_RATIO"], errors="coerce"),
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["Prime", "Balanced", "Leveraged", "Critical"],
    )

    if "AMT_CREDIT" in segmented.columns:
        fig = px.box(
            segmented.dropna(subset=["Risk Segment"]),
            x="Risk Segment",
            y="AMT_CREDIT",
            color=TARGET if TARGET in segmented.columns else None,
            title="Credit Exposure Distribution by Customer Segment",
        )

        fig.update_layout(yaxis_tickformat=",")
        plot_chart(apply_plot_theme(fig), key="risk_segment_box")
    else:
        st.info("AMT_CREDIT column is missing, so exposure box plot cannot be shown.")


def show_xai(model, metadata: dict, reference: pd.DataFrame) -> None:
    luxury_hero(
        "Explainable AI",
        "Convert model output into transparent, auditable, and regulator-friendly risk narratives.",
        "Model Transparency",
    )

    feature_names = get_feature_names(metadata)

    try:
        row = preprocess_input(reference.iloc[[0]].copy(), feature_names)
        contributions, explanation = shap_contributions(model, row)
    except Exception as exc:
        st.error(f"SHAP explanation could not be generated: {exc}")
        return

    summary_path = ROOT / "outputs" / "figures" / "shap_summary_plot.png"
    waterfall_path = ROOT / "outputs" / "figures" / "shap_waterfall_plot.png"

    c1, c2 = st.columns(2)

    with c1:
        section("SHAP Summary Plot")

        if summary_path.exists():
            show_image(str(summary_path))
        else:
            st.info("Run model training to generate SHAP summary plot.")

    with c2:
        section("SHAP Waterfall Plot")

        if waterfall_path.exists():
            show_image(str(waterfall_path))
        else:
            st.info("Run model training to generate SHAP waterfall plot.")

    section("Local Explanation")
    st.write(explanation)

    if contributions is None or contributions.empty:
        st.info("No local contribution values were returned.")
        return

    if "shap_value" not in contributions.columns:
        st.info("SHAP contribution table does not contain a shap_value column.")
        show_df(contributions, key="xai_all_contributions")
        return

    positive = (
        contributions[contributions["shap_value"] > 0]
        .sort_values("shap_value", ascending=False)
    )

    negative = (
        contributions[contributions["shap_value"] < 0]
        .sort_values("shap_value", ascending=True)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk-Increasing Drivers")
        if positive.empty:
            st.info("No risk-increasing drivers found.")
        else:
            show_df(positive, key="xai_positive")

    with col2:
        st.subheader("Risk-Reducing Drivers")
        if negative.empty:
            st.info("No risk-reducing drivers found.")
        else:
            show_df(negative, key="xai_negative")


def show_model_performance(metadata: dict) -> None:
    luxury_hero(
        "Model Performance",
        "Review model comparison, classification diagnostics, ROC curve, confusion matrix, and feature importance.",
        "Model Validation Suite",
    )

    metrics = pd.DataFrame(metadata.get("metrics", []))

    if metrics.empty:
        st.warning("No metrics found. Re-run training to generate model comparison.")
        return

    section("Model Leaderboard")
    show_df(metrics, key="model_metrics", height=300)

    best = metrics.iloc[0]
    metric_cols = st.columns(5)

    for column, col in zip(
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        metric_cols,
    ):
        with col:
            if column in best and pd.notna(best[column]):
                luxury_metric(column, f"{float(best[column]):.3f}", "best model")
            else:
                luxury_metric(column, "N/A", "not found")

    c1, c2 = st.columns(2)

    with c1:
        plot_chart(confusion_matrix_figure(metadata), key="confusion_matrix")

    with c2:
        plot_chart(roc_curve_figure(metadata), key="roc_curve")

    section("Top Model Drivers")

    try:
        plot_chart(feature_importance_figure(FEATURE_IMPORTANCE_PATH), key="feature_importance")
    except Exception as exc:
        st.info(f"Feature importance chart could not be loaded: {exc}")


def show_business_insights() -> None:
    luxury_hero(
        "Business Insights",
        "Executive recommendations for credit policy, underwriting operations, monitoring, and customer-risk strategy.",
        "Actionable Banking Strategy",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        luxury_info_card(
            "High-Risk Pattern",
            "High-risk applications usually show stretched credit-income ratios, weak affordability, unstable employment history, and larger annuity burden.",
        )

    with c2:
        luxury_info_card(
            "Low-Risk Pattern",
            "Low-risk applications generally show stable income, lower debt burden, stronger external scores, and manageable repayment structure.",
        )

    with c3:
        luxury_info_card(
            "Operating Policy",
            "Use the model as a premium decision-support layer. Final approvals should remain controlled by policy rules and human review.",
        )

    section("Recommended Banking Actions")

    st.markdown(
        """
        - Apply **risk-based pricing** for medium-risk applications instead of rejecting all borderline cases.
        - Route **high-risk applications** to enhanced affordability review and manual underwriting.
        - Monitor **model drift monthly** and recalibrate score thresholds quarterly.
        - Use **SHAP explanations** to support transparent decision narratives.
        - Combine model probability with **hard policy rules**, compliance checks, and adverse-action governance.
        - Track approval rate, default rate, profitability, and fairness indicators as executive KPIs.
        """
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    try:
        model, metadata = cached_artifacts()
        data = cached_data()
        reference = cached_reference()

    except FileNotFoundError as exc:
        st.error(f"Required project file was not found: {exc}")
        st.stop()

    except Exception as exc:
        st.error(f"Project artifacts could not be loaded: {exc}")
        st.stop()

    page = sidebar_nav()

    try:
        if page == "Dashboard":
            show_dashboard(model, data, metadata)

        elif page == "Customer Prediction":
            show_prediction(model, metadata, reference)

        elif page == "Risk Analysis":
            show_risk_analysis(model, data, metadata)

        elif page == "Explainable AI":
            show_xai(model, metadata, reference)

        elif page == "Model Performance":
            show_model_performance(metadata)

        elif page == "Business Insights":
            show_business_insights()

        else:
            st.error("Unknown navigation option selected.")

    except Exception as exc:
        st.error(f"Dashboard page failed to render: {exc}")
        st.info(
            "Check that data files, model artifacts, and metadata are available in the expected folders."
        )


if __name__ == "__main__":
    main()