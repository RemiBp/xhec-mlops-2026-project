from pathlib import Path

from prefect.testing.utilities import prefect_test_harness

from src.modelling.utils import load_artifact
from src.modelling.workflows import build_deployment, retraining_flow


def test_schedule_is_weekly_in_paris():
    deployment = build_deployment()
    assert deployment.schedule.cron == "0 9 * * 1"
    assert deployment.schedule.timezone == "Europe/Paris"


def test_real_prefect_flow_publishes_reloadable_model(training_csv, tmp_path):
    model = tmp_path / "flow-model.pkl"
    with prefect_test_harness():
        result = retraining_flow(str(training_csv), str(model), None, None)
    assert model.exists()
    assert load_artifact(Path(model))["metadata"]["model_version"] == result["model_version"]
