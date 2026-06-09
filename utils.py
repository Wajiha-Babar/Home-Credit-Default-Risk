"""Shared data-preparation utilities for the Home Credit project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
TARGET = "TARGET"
DAYS_EMPLOYED_PLACEHOLDER = 365243


def configure_logging() -> None:
    """Configure project logging once per process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_application_data(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load Home Credit application data with predictable missing-value handling."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path, nrows=nrows)


def safe_divide(numerator: Any, denominator: Any) -> Any:
    """Divide while converting zero denominators to missing values."""
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    elif denominator in (0, None) or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def clean_application_data(
    df: pd.DataFrame,
    target: str = TARGET,
    missing_threshold: float = 0.60,
    clip_outliers: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Clean raw application data and return cleaned data plus dropped columns."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    if "DAYS_EMPLOYED" in cleaned.columns:
        cleaned["DAYS_EMPLOYED"] = cleaned["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_PLACEHOLDER, np.nan)

    missing_pct = cleaned.isna().mean()
    dropped_columns = [col for col in missing_pct[missing_pct > missing_threshold].index if col != target]
    cleaned = cleaned.drop(columns=dropped_columns)

    if clip_outliers:
        numeric_columns = [col for col in cleaned.select_dtypes(include=np.number).columns if col != target]
        for column in numeric_columns:
            lower, upper = cleaned[column].quantile([0.01, 0.99])
            cleaned[column] = cleaned[column].clip(lower, upper)

    return cleaned, dropped_columns


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the project feature-engineering variables."""
    data = df.copy()
    required_defaults = {
        "AMT_CREDIT": np.nan,
        "AMT_INCOME_TOTAL": np.nan,
        "AMT_ANNUITY": np.nan,
        "DAYS_EMPLOYED": np.nan,
        "DAYS_BIRTH": np.nan,
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 1,
    }
    for column, default in required_defaults.items():
        if column not in data.columns:
            data[column] = default

    data["CREDIT_INCOME_RATIO"] = safe_divide(data["AMT_CREDIT"], data["AMT_INCOME_TOTAL"])
    data["ANNUITY_INCOME_RATIO"] = safe_divide(data["AMT_ANNUITY"], data["AMT_INCOME_TOTAL"])
    data["EMPLOYMENT_AGE_RATIO"] = safe_divide(data["DAYS_EMPLOYED"], data["DAYS_BIRTH"])
    data["CREDIT_TERM"] = safe_divide(data["AMT_ANNUITY"], data["AMT_CREDIT"])
    data["CHILDREN_RATIO"] = safe_divide(data["CNT_CHILDREN"], data["CNT_FAM_MEMBERS"])
    return data.replace([np.inf, -np.inf], np.nan)


def align_columns(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Add missing columns, remove extras, and enforce training feature order."""
    aligned = df.copy()
    missing_columns = [column for column in feature_names if column not in aligned.columns]
    if missing_columns:
        LOGGER.info("Adding %d missing prediction columns.", len(missing_columns))
        aligned = pd.concat(
            [aligned, pd.DataFrame({column: [np.nan] * len(aligned) for column in missing_columns}, index=aligned.index)],
            axis=1,
        )
    return aligned.loc[:, feature_names].copy()


def prepare_features(
    df: pd.DataFrame,
    target: str = TARGET,
    missing_threshold: float = 0.60,
    clip_outliers: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None, dict[str, Any]]:
    """Clean data, engineer features, and split X/y for training."""
    cleaned, dropped_columns = clean_application_data(df, target, missing_threshold, clip_outliers)
    prepared = add_engineered_features(cleaned)
    y = prepared[target].astype(int) if target in prepared.columns else None
    X = prepared.drop(columns=[target], errors="ignore")
    metadata = {
        "feature_names": X.columns.tolist(),
        "dropped_missing_columns": dropped_columns,
        "numeric_features": X.select_dtypes(include=np.number).columns.tolist(),
        "categorical_features": X.select_dtypes(include=["object", "category"]).columns.tolist(),
    }
    return X, y, metadata


def preprocess_input(raw_input: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Prepare one or more dashboard input rows for model prediction."""
    prepared = raw_input.copy()
    if "DAYS_EMPLOYED" in prepared.columns:
        prepared["DAYS_EMPLOYED"] = prepared["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_PLACEHOLDER, np.nan)
    prepared = add_engineered_features(prepared)
    return align_columns(prepared, feature_names)


def integer_days_from_years(years: int | float) -> int:
    """Convert years to negative integer days without assigning floats to int columns."""
    return int(round(-float(years) * 365.25))
