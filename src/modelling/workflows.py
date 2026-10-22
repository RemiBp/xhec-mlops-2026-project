"""Run or serve weekly retraining with the same tested training functions."""

import argparse
from pathlib import Path

from prefect import flow, serve, task
from prefect.client.schemas.schedules import CronSchedule

from src.modelling.main import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    ROOT,
    build_model_bundle,
    publish_model_bundle,
)


@task(name="fit-and-evaluate")
def fit_and_evaluate(data_path: str) -> dict:
    """Select on validation and evaluate the chosen model on the held-out test."""
    return build_model_bundle(Path(data_path))


@task(name="publish-model", retries=1, retry_delay_seconds=1)
def publish_model(
    bundle: dict, model_path: str, tracking_path: str | None, report_path: str | None
):
    """Publish a complete artifact, preserving the previous one until success."""
    return publish_model_bundle(
        bundle,
        Path(model_path),
        Path(tracking_path) if tracking_path else None,
        Path(report_path) if report_path else None,
    )


@flow(name="abalone-retraining", log_prints=True)
def retraining_flow(
    data_path: str = str(DEFAULT_DATA),
    model_path: str = str(DEFAULT_MODEL),
    tracking_path: str | None = str(ROOT / "mlruns"),
    report_path: str | None = str(ROOT / "reports/metrics.json"),
) -> dict:
    bundle = fit_and_evaluate(data_path)
    result = publish_model(bundle, model_path, tracking_path, report_path)
    print(f"Published {result['model_version']}; test RMSE {result['test']['rmse']:.3f} rings")
    return result


def build_deployment():
    """Define Monday 09:00 retraining, respecting daylight saving in Paris."""
    return retraining_flow.to_deployment(
        name="weekly",
        schedule=CronSchedule(cron="0 9 * * 1", timezone="Europe/Paris"),
        tags=["abalone"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "serve"])
    args = parser.parse_args()
    if args.action == "run":
        retraining_flow()
    else:
        serve(build_deployment())
