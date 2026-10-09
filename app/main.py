from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store["pipeline"] = joblib.load(MODEL_PATH)
    yield
    model_store.clear()


app = FastAPI(
    title="Horse Racing Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


class Runner(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runner_id: str
    features: dict[str, float | None]


class RaceRequest(BaseModel):
    race_id: str
    runners: list[Runner] = Field(min_length=2)


@app.get("/")
def home():
    return {"message": "Horse Racing API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: RaceRequest):
    pipe = model_store["pipeline"]

    # Retrieve the exact feature names used during training.
    if not hasattr(pipe, "feature_names_in_"):
        raise HTTPException(
            status_code=500,
            detail="Saved pipeline does not contain feature names.",
        )

    expected_features = list(pipe.feature_names_in_)

    # Validate that every runner has exactly the expected features.
    for runner in request.runners:
        supplied = set(runner.features)
        expected = set(expected_features)

        if supplied != expected:
            raise HTTPException(
                status_code=422,
                detail={
                    "runner_id": runner.runner_id,
                    "missing_features": sorted(expected - supplied),
                    "unexpected_features": sorted(supplied - expected),
                },
            )

    X = pd.DataFrame(
        [runner.features for runner in request.runners],
        columns=expected_features,
    )

    try:
        probabilities = pipe.predict_proba(X)[:, 1]

        total = probabilities.sum()

        if not np.isfinite(total) or total <= 0:
            raise ValueError("Invalid probability total.")

        normalised = probabilities / total

    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "race_id": request.race_id,
        "predictions": [
            {
                "runner_id": runner.runner_id,
                "win_probability": float(prob),
            }
            for runner, prob in zip(request.runners, normalised)
        ],
    }