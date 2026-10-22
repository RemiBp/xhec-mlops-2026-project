"""Serve ring counts from the same fitted pipeline used during evaluation."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.modelling.config import DEFAULT_MODEL
from src.modelling.predicting import predict_records
from src.web_service.lib.inference import ModelStore
from src.web_service.lib.models import Abalone, Batch, BatchPredictions, Prediction


def create_app(model_path: Path | None = None) -> FastAPI:
    store = ModelStore(model_path or Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL))))

    @asynccontextmanager
    async def lifespan(app):
        store.get()
        yield

    app = FastAPI(title="Abalone ring prediction", version="0.1.0", lifespan=lifespan)

    def current_bundle():
        try:
            return store.get()
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=503, detail="Model artifact unavailable") from exc

    @app.get("/")
    def home():
        return {"service": "Abalone ring prediction", "target": "rings", "docs": "/docs"}

    @app.get("/health")
    def health():
        metadata = current_bundle()["metadata"]
        return {
            "status": "ok",
            "model_version": metadata["model_version"],
            "trained_at": metadata["trained_at"],
        }

    def predict_batch(records):
        bundle = current_bundle()
        values = predict_records(bundle, [record.dict() for record in records])
        return [
            Prediction(
                rings=value,
                approximate_age_years=value + 1.5,
                model_version=bundle["metadata"]["model_version"],
            )
            for value in values
        ]

    @app.post("/predict", response_model=Prediction)
    def predict(payload: Abalone):
        return predict_batch([payload])[0]

    @app.post("/predict/batch", response_model=BatchPredictions)
    def batch_predict(payload: Batch):
        return BatchPredictions(predictions=predict_batch(payload.records))

    return app


app = create_app()
