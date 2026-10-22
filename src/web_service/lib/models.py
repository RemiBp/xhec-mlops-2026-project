"""The API accepts the same eight measurements used during training."""

from typing import Literal

from pydantic import BaseModel, confloat, conlist, validator

Measurement = confloat(gt=0, allow_inf_nan=False)


class Abalone(BaseModel):
    sex: Literal["M", "F", "I"]
    length: Measurement
    diameter: Measurement
    height: Measurement
    whole_weight: Measurement
    shucked_weight: Measurement
    viscera_weight: Measurement
    shell_weight: Measurement

    @validator("*", pre=True)
    def measurements_are_numbers(cls, value, field):
        if field.name != "sex" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("Measurements must be JSON numbers")
        return value

    class Config:
        extra = "forbid"
        schema_extra = {
            "example": {
                "sex": "M",
                "length": 0.455,
                "diameter": 0.365,
                "height": 0.095,
                "whole_weight": 0.514,
                "shucked_weight": 0.2245,
                "viscera_weight": 0.101,
                "shell_weight": 0.15,
            }
        }


class Batch(BaseModel):
    records: conlist(Abalone, min_items=1, max_items=1000)

    class Config:
        extra = "forbid"


class Prediction(BaseModel):
    rings: confloat(allow_inf_nan=False)
    approximate_age_years: confloat(allow_inf_nan=False)
    model_version: str


class BatchPredictions(BaseModel):
    predictions: list[Prediction]
