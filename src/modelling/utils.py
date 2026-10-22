"""Persist a complete fitted pipeline using an atomic replacement."""

import tempfile
from pathlib import Path

import joblib


def save_artifact(bundle: dict, path: Path) -> Path:
    """Publish the model only after serialization completes successfully."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        joblib.dump(bundle, pending, compress=3)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    return path


def load_artifact(path: Path) -> dict:
    """Load only an artifact produced locally by this project."""
    return joblib.load(path)
