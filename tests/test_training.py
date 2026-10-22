import numpy as np
import pandas as pd
import pytest

from src.modelling.predicting import predict_records
from src.modelling.preprocessing import FEATURES, load_data, split_data


def test_split_is_disjoint_complete_and_repeatable(training_csv):
    data, _ = load_data(training_csv)
    train, validation, test = split_data(data)
    assert [len(train), len(validation), len(test)] == [60, 20, 20]
    groups = [set(part.index) for part in (train, validation, test)]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert set.union(*groups) == set(data.index)
    assert list(split_data(data)[0].index) == list(train.index)
    assert "rings" not in FEATURES


def test_invalid_measurements_are_excluded(training_csv):
    raw = pd.read_csv(training_csv)
    raw.loc[0, "Height"] = 0
    raw.loc[1, "Length"] = np.inf
    raw.to_csv(training_csv, index=False)
    clean, counts = load_data(training_csv)
    assert counts["excluded_rows"] == 2
    assert len(clean) == 98


def test_target_cannot_be_used_as_an_input(training_csv):
    raw = pd.read_csv(training_csv)
    raw["Age"] = raw.Rings + 1.5
    raw.to_csv(training_csv, index=False)
    with pytest.raises(ValueError, match="eight features"):
        load_data(training_csv)


def test_saved_pipeline_matches_training_predictions(fitted_bundle, training_csv):
    _, bundle = fitted_bundle
    frame, _ = load_data(training_csv)
    records = frame[FEATURES].head(3).to_dict(orient="records")
    np.testing.assert_allclose(
        predict_records(bundle, records), bundle["pipeline"].predict(frame[FEATURES].head(3))
    )
    metadata = bundle["metadata"]
    assert metadata["model"] == min(
        metadata["validation"], key=lambda key: metadata["validation"][key]["rmse"]
    )
    assert metadata["test_rows"] == 20
    assert np.isfinite(metadata["test"]["rmse"])


def test_prediction_rejects_target_or_missing_feature(fitted_bundle):
    _, bundle = fitted_bundle
    with pytest.raises(ValueError, match="exactly"):
        predict_records(bundle, [{"sex": "M", "rings": 10}])
