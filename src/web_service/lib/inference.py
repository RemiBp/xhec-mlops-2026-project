"""Reload an atomically replaced model on the next prediction request."""

from pathlib import Path
from threading import Lock

import sklearn

from src.modelling.preprocessing import FEATURES
from src.modelling.utils import load_artifact


class ModelStore:
    def __init__(self, path: Path):
        self.path = path
        self.signature = None
        self.bundle = None
        self.lock = Lock()

    def get(self) -> dict:
        with self.lock:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            if signature != self.signature:
                bundle = load_artifact(self.path)
                metadata = bundle["metadata"]
                if metadata["features"] != FEATURES:
                    raise ValueError("Model feature contract does not match the API")
                if metadata["sklearn_version"] != sklearn.__version__:
                    raise ValueError("Model and API use different scikit-learn versions")
                self.bundle, self.signature = bundle, signature
            return self.bundle
