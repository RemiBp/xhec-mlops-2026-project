"""Project paths shared by training and inference without importing MLflow."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/abalone.csv"
DEFAULT_MODEL = ROOT / "src/web_service/local_objects/model.pkl"
