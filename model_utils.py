"""Model training, scoring, artifact, and SHAP helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def build_preprocessors(X: pd.DataFrame) -> tuple[ColumnTransformer, ColumnTransformer, list[str], list[str], list[str]]:
    """Create scaled and tree preprocessors with stable categorical handling."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    binary_cats = [col for col in categorical_features if X[col].nunique(dropna=True) == 2]
    multi_cats = [col for col in categorical_features if X[col].nunique(dropna=True) > 2]

    binary_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
        ]
    )
    multi_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=50)),
        ]
    )
    numeric_scaled = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    numeric_tree = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    preprocessor_scaled = ColumnTransformer(
        [("num", numeric_scaled, numeric_features), ("bin", binary_transformer, binary_cats), ("cat", multi_transformer, multi_cats)],
        verbose_feature_names_out=False,
    )
    preprocessor_tree = ColumnTransformer(
        [("num", numeric_tree, numeric_features), ("bin", binary_transformer, binary_cats), ("cat", multi_transformer, multi_cats)],
        verbose_feature_names_out=False,
    )
    return preprocessor_scaled, preprocessor_tree, numeric_features, categorical_features, binary_cats + multi_cats


def create_model_candidates(X: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Build tuned candidate estimators using class weights and SMOTE where appropriate."""
    preprocessor_scaled, preprocessor_tree, *_ = build_preprocessors(X)
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    candidates: dict[str, Any] = {
        "Logistic Regression + SMOTE": ImbPipeline(
            [
                ("preprocess", preprocessor_scaled),
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                ("model", LogisticRegression(max_iter=800, class_weight="balanced", solver="lbfgs", random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", preprocessor_tree),
                ("model", RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
            ]
        ),
    }
    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = Pipeline(
            [
                ("preprocess", preprocessor_tree),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="auc",
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        n_jobs=1,
                        tree_method="hist",
                    ),
                ),
            ]
        )
    return candidates


def parameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Small but useful tuning grids that finish on a laptop."""
    return {
        "Logistic Regression + SMOTE": {"model__C": [0.5, 1.0]},
        "Random Forest": {
            "model__n_estimators": [180],
            "model__max_depth": [12, 16],
            "model__min_samples_leaf": [25],
        },
        "XGBoost": {
            "model__n_estimators": [260],
            "model__max_depth": [3, 4],
            "model__learning_rate": [0.05],
            "model__subsample": [0.85],
            "model__colsample_bytree": [0.85],
        },
    }


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Return standard binary classification metrics."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC AUC": float(roc_auc_score(y_true, y_proba)),
    }


def train_compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path = "outputs",
    sample_size: int | None = 70000,
    tune: bool = True,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Train LR, RF, XGBoost, compare with CV, and save the best model."""
    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    models_dir = output_path / "models"
    reports_dir = output_path / "reports"
    for directory in [figures_dir, models_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if sample_size and len(X) > sample_size:
        _, X, _, y = train_test_split(X, y, test_size=sample_size / len(X), random_state=42, stratify=y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    candidates = create_model_candidates(X_train, y_train)
    grids = parameter_grids()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fitted: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, estimator in candidates.items():
        LOGGER.info("Training %s", name)
        if tune and name in grids:
            search_jobs = 1
            search = GridSearchCV(estimator, grids[name], scoring="roc_auc", cv=cv, n_jobs=search_jobs, error_score="raise")
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
            cv_auc = float(search.best_score_)
        else:
            model = estimator.fit(X_train, y_train)
            best_params = {}
            cv_auc = float(cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1).mean())

        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, pred, proba)
        rows.append({"Model": name, "CV ROC AUC": cv_auc, **metrics, "Best Params": json.dumps(best_params)})
        fitted[name] = {"model": model, "pred": pred, "proba": proba}

    comparison = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False)
    best_name = str(comparison.iloc[0]["Model"])
    best_model = fitted[best_name]["model"]

    joblib.dump(best_model, models_dir / "best_model.joblib")
    metadata = {
        "best_model_name": best_name,
        "target": "TARGET",
        "features": X.columns.tolist(),
        "metrics": comparison.to_dict(orient="records"),
        "test_confusion_matrix": confusion_matrix(y_test, fitted[best_name]["pred"]).tolist(),
        "test_roc_curve": _roc_payload(y_test, fitted[best_name]["proba"]),
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    comparison.to_csv(reports_dir / "model_metrics.csv", index=False)
    (reports_dir / "metrics_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _save_feature_importance(best_model, figures_dir / "feature_importance_top_20.csv")
    return best_model, comparison, metadata


def _roc_payload(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, list[float]]:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


def _save_feature_importance(model: Any, path: Path) -> None:
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            values = np.abs(estimator.coef_[0])
        else:
            return
        pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False).head(30).to_csv(path, index=False)
    except Exception as exc:
        LOGGER.warning("Could not save feature importance: %s", exc)


def load_artifacts(model_path: str | Path, metadata_path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load model and metadata with clear errors."""
    model_file = Path(model_path)
    metadata_file = Path(metadata_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata artifact not found: {metadata_file}")
    return joblib.load(model_file), json.loads(metadata_file.read_text(encoding="utf-8"))


def predict_risk(model: Any, X: pd.DataFrame) -> tuple[int, float, str, float]:
    """Predict class, probability, category, and confidence."""
    probability = float(model.predict_proba(X)[0, 1])
    predicted_class = int(probability >= 0.5)
    if probability < 0.20:
        category = "Low Risk"
    elif probability < 0.50:
        category = "Medium Risk"
    else:
        category = "High Risk"
    confidence = float(max(probability, 1 - probability))
    return predicted_class, probability, category, confidence


def shap_contributions(model: Any, X: pd.DataFrame, max_features: int = 10) -> tuple[pd.DataFrame, str]:
    """Return local SHAP contributions for the positive default class."""
    try:
        import shap

        transformed = model.named_steps["preprocess"].transform(X)
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        estimator = model.named_steps["model"]
        explainer = shap.TreeExplainer(estimator)
        values = explainer(transformed).values
        if getattr(values, "ndim", 0) == 3:
            values = values[:, :, 1]
        local = pd.Series(values[0], index=feature_names).sort_values()
        contributions = pd.concat([local.head(max_features), local.tail(max_features)]).sort_values()
        explanation = _natural_language_explanation(local)
        return contributions.rename("shap_value").reset_index().rename(columns={"index": "feature"}), explanation
    except Exception as exc:
        LOGGER.warning("SHAP explanation failed: %s", exc)
        return pd.DataFrame(columns=["feature", "shap_value"]), "SHAP explanation is unavailable for the current model artifact."


def _natural_language_explanation(local_values: pd.Series) -> str:
    positive = local_values.sort_values(ascending=False).head(3).index.tolist()
    negative = local_values.sort_values().head(3).index.tolist()
    return (
        "The customer's risk increased primarily due to "
        + ", ".join(positive)
        + ". Risk was reduced by "
        + ", ".join(negative)
        + "."
    )
