"""Compare a baseline and two small models without selecting on test scores."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.modelling.preprocessing import FEATURES, NUMERIC_FEATURES, SEED, TARGET


def make_pipeline(name: str) -> Pipeline:
    """Fit numeric scaling and categorical encoding inside each model pipeline."""
    estimators = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=SEED, n_jobs=1
        ),
    }
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex"]),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimators[name])])


def metrics(y, predictions) -> dict:
    """Report RMSE and MAE in rings, plus R squared."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }


def select_model(train, validation):
    """Choose by validation RMSE and refit only the winner on development data."""
    scores = {}
    for name in ("mean_baseline", "linear_regression", "random_forest"):
        pipeline = make_pipeline(name).fit(train[FEATURES], train[TARGET])
        scores[name] = metrics(validation[TARGET], pipeline.predict(validation[FEATURES]))
    name = min(scores, key=lambda key: scores[key]["rmse"])
    development = pd.concat([train, validation]).sort_index()
    fitted = make_pipeline(name).fit(development[FEATURES], development[TARGET])
    return fitted, name, scores
