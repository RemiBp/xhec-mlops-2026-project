"""Predict from a complete saved preprocessing/model pipeline."""

import numpy as np
import pandas as pd

from src.modelling.preprocessing import FEATURES


def predict_records(bundle: dict, records: list[dict]) -> list[float]:
    """Require exactly the training inputs and return finite ring predictions."""
    if not records or any(set(row) != set(FEATURES) for row in records):
        raise ValueError("Each record must contain exactly the eight training features")
    predictions = bundle["pipeline"].predict(pd.DataFrame(records)[FEATURES])
    if not np.isfinite(predictions).all():
        raise ValueError("Model returned non-finite predictions")
    return predictions.tolist()
