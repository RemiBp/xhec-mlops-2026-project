"""Train, evaluate and save the Abalone model from a CSV file."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import joblib
import mlflow
import sklearn

from src.modelling.config import DEFAULT_DATA, DEFAULT_MODEL, ROOT
from src.modelling.preprocessing import FEATURES, SEED, TARGET, load_data, split_data
from src.modelling.training import metrics, select_model
from src.modelling.utils import save_artifact


def build_model_bundle(data_path: Path) -> dict:
    """Use validation for selection; evaluate the selected model on test once."""
    data, counts = load_data(data_path)
    train, validation, test = split_data(data)
    model, name, validation_scores = select_model(train, validation)
    test_scores = metrics(test[TARGET], model.predict(test[FEATURES]))
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    serialized_model = BytesIO()
    joblib.dump(model, serialized_model)
    version = hashlib.sha256(serialized_model.getvalue()).hexdigest()[:12]
    metadata = {
        **counts,
        "model": name,
        "model_version": version,
        "seed": SEED,
        "features": FEATURES,
        "target": TARGET,
        "units": "rings",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "dataset_sha256": dataset_hash,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "validation": validation_scores,
        "test": test_scores,
    }
    return {"pipeline": model, "metadata": metadata}


def publish_model_bundle(bundle, model_path, tracking_path, report_path) -> dict:
    """Record experiments and publish the already evaluated model."""
    metadata = bundle["metadata"]
    if tracking_path is not None:
        mlflow.set_tracking_uri(tracking_path.resolve().as_uri())
        mlflow.set_experiment("abalone-training")
        for candidate, scores in metadata["validation"].items():
            with mlflow.start_run(run_name=candidate):
                mlflow.log_params(
                    {"model": candidate, "seed": SEED, "dataset_sha256": metadata["dataset_sha256"]}
                )
                mlflow.log_metrics({f"validation_{key}": value for key, value in scores.items()})
                if candidate == metadata["model"]:
                    mlflow.log_metrics(
                        {f"test_{key}": value for key, value in metadata["test"].items()}
                    )
    save_artifact(bundle, model_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def train_pipeline(
    data_path: Path = DEFAULT_DATA,
    model_path: Path = DEFAULT_MODEL,
    tracking_path: Path | None = ROOT / "mlruns",
    report_path: Path | None = ROOT / "reports/metrics.json",
) -> dict:
    """Run the same build and publish steps without needing a Prefect server."""
    return publish_model_bundle(
        build_model_bundle(data_path), model_path, tracking_path, report_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(train_pipeline(args.data, args.model), indent=2))
