"""Train and compare Home Credit default-risk models.

Run:
    python train_models.py
"""

from __future__ import annotations

from pathlib import Path
import json

from model_utils import train_compare_models
from utils import configure_logging, load_application_data, prepare_features


def main() -> None:
    configure_logging()
    data_path = Path("data/application_train.csv")
    df = load_application_data(data_path)
    X, y, feature_metadata = prepare_features(df)
    if y is None:
        raise ValueError("TARGET column is required for training.")
    best_model, comparison, metadata = train_compare_models(X, y, output_dir="outputs", sample_size=45000, tune=True)
    metadata.update(feature_metadata)
    (Path("outputs") / "models" / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (Path("outputs") / "reports" / "metrics_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Best model:", metadata["best_model_name"])
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
