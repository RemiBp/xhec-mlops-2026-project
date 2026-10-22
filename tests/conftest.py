"""Small, separate datasets for fast tests without network access."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def training_csv(tmp_path):
    random = np.random.default_rng(42)
    size = 100
    frame = pd.DataFrame(
        {
            "Sex": np.tile(["M", "F", "I", "M"], 25),
            "Length": random.uniform(0.3, 0.7, size),
            "Diameter": random.uniform(0.2, 0.5, size),
            "Height": random.uniform(0.05, 0.2, size),
            "Whole weight": random.uniform(0.2, 1.5, size),
            "Shucked weight": random.uniform(0.1, 0.8, size),
            "Viscera weight": random.uniform(0.05, 0.4, size),
            "Shell weight": random.uniform(0.1, 0.6, size),
        }
    )
    frame["Rings"] = np.round(5 + frame["Length"] * 10 + random.normal(0, 1, size)).astype(int)
    path = tmp_path / "abalone.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def fitted_bundle(training_csv, tmp_path):
    from src.modelling.main import train_pipeline
    from src.modelling.utils import load_artifact

    path = tmp_path / "model.pkl"
    train_pipeline(training_csv, path, tracking_path=None, report_path=None)
    return path, load_artifact(path)
