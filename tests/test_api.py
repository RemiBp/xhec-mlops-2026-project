from copy import deepcopy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.modelling.predicting import predict_records
from src.modelling.utils import save_artifact
from src.web_service.lib.models import Abalone
from src.web_service.main import create_app

RECORD = Abalone.Config.schema_extra["example"]


def test_http_prediction_and_batch_match_saved_model(fitted_bundle):
    path, bundle = fitted_bundle
    with TestClient(create_app(path)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json=RECORD)
        assert response.status_code == 200
        result = response.json()
        np.testing.assert_allclose([result["rings"]], predict_records(bundle, [RECORD]))
        assert result["approximate_age_years"] == result["rings"] + 1.5
        batch = client.post("/predict/batch", json={"records": [RECORD, RECORD]})
        assert batch.json()["predictions"] == [result, result]
        assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize(
    "changes",
    [
        {"sex": "X"},
        {"height": 0},
        {"length": -1},
        {"rings": 8},
        {"age": 9},
        {"whole_weight": None},
        {"length": True},
        {"height": "0.1"},
    ],
)
def test_invalid_inputs_are_rejected(fitted_bundle, changes):
    with TestClient(create_app(fitted_bundle[0])) as client:
        assert client.post("/predict", json={**RECORD, **changes}).status_code == 422


def test_missing_input_and_empty_batch_are_rejected(fitted_bundle):
    with TestClient(create_app(fitted_bundle[0])) as client:
        assert client.post("/predict", json={"sex": "M"}).status_code == 422
        assert client.post("/predict/batch", json={"records": []}).status_code == 422
        assert client.post("/predict/batch", json={"records": [RECORD] * 1001}).status_code == 422


def test_new_artifact_is_loaded_without_restarting_api(fitted_bundle):
    path, bundle = fitted_bundle
    with TestClient(create_app(path)) as client:
        original = client.get("/health").json()["model_version"]
        replacement = deepcopy(bundle)
        replacement["metadata"]["model_version"] = "replacement-test"
        save_artifact(replacement, path)
        assert original != client.post("/predict", json=RECORD).json()["model_version"]


def test_no_ready_service_without_model(tmp_path):
    with pytest.raises(FileNotFoundError), TestClient(create_app(tmp_path / "absent.pkl")):
        pass
