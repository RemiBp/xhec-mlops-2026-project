"""Check the running HTTP service against the locally trained model."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.modelling.config import DEFAULT_MODEL
from src.modelling.predicting import predict_records
from src.modelling.utils import load_artifact
from src.web_service.lib.models import Abalone


def call(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(base + path, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def smoke(base: str, model: Path):
    for _ in range(90):
        try:
            health = call(base, "/health")
            break
        except (OSError, URLError):
            time.sleep(1)
    else:
        raise RuntimeError("API did not become ready")
    record = Abalone.Config.schema_extra["example"]
    prediction = call(base, "/predict", record)
    expected = predict_records(load_artifact(model), [record])[0]
    assert abs(prediction["rings"] - expected) < 1e-7
    batch = call(base, "/predict/batch", {"records": [record, record]})
    assert batch["predictions"] == [prediction, prediction]
    assert prediction["approximate_age_years"] == prediction["rings"] + 1.5
    assert "/predict" in call(base, "/openapi.json")["paths"]
    try:
        call(base, "/predict", {**record, "height": 0})
    except HTTPError as exc:
        assert exc.code == 422
    else:
        raise AssertionError("Invalid measurement was accepted")
    result = {
        "health": health,
        "single_batch_and_local_parity": True,
        "invalid_input_status": 422,
        "openapi": True,
        "prediction": prediction,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/http_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8526")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    smoke(args.base, args.model)
