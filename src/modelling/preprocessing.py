"""Keep the feature contract and split in one place."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
FEATURES = ["sex"] + NUMERIC_FEATURES
TARGET = "rings"
SEED = 42


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the Kaggle CSV and exclude physically invalid or incomplete rows."""
    raw = pd.read_csv(path)
    data = raw.rename(columns=lambda name: name.lower().replace(" ", "_"))
    if set(data.columns) != set(FEATURES + [TARGET]):
        raise ValueError("Dataset must contain the eight features and Rings, exactly once")
    for name in NUMERIC_FEATURES + [TARGET]:
        data[name] = pd.to_numeric(data[name], errors="raise")
    valid = data.sex.isin(["M", "F", "I"])
    valid &= np.isfinite(data[NUMERIC_FEATURES + [TARGET]]).all(axis=1)
    valid &= data[NUMERIC_FEATURES + [TARGET]].gt(0).all(axis=1)
    valid &= data[TARGET].mod(1).eq(0)
    clean = data.loc[valid].copy()
    if len(clean) < 30:
        raise ValueError("At least 30 valid rows are required for a three-way split")
    return clean, {
        "source_rows": len(data),
        "excluded_rows": int((~valid).sum()),
        "valid_rows": len(clean),
    }


def split_data(data: pd.DataFrame):
    """Return disjoint 60/20/20 training, validation and test partitions."""
    development, test = train_test_split(data, test_size=0.2, random_state=SEED)
    train, validation = train_test_split(development, test_size=0.25, random_state=SEED)
    return train, validation, test
