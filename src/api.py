"""FastAPI model-serving API with Prometheus metrics.

Endpoints:
  GET  /health   - liveness/readiness check
  POST /predict  - predict the Iris species from 4 measurements
  GET  /metrics  - Prometheus metrics (scraped by Prometheus, shown in Grafana)
Run:  uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import json
import time

import joblib
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.train import METRICS_PATH, MODEL_PATH, SPECIES, train

REQUESTS = Counter("iris_requests_total", "Total HTTP requests", ["endpoint", "status"])
PREDICTIONS = Counter("iris_predictions_total", "Predictions made per species", ["species"])
LATENCY = Histogram("iris_request_latency_seconds", "Prediction latency in seconds", ["endpoint"])
MODEL_ACCURACY = Gauge("iris_model_accuracy", "Test accuracy of the loaded model")

app = FastAPI(title="Iris MLOps API", version="1.0.0")

# Load the trained model (train on the fly if the file is missing).
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
else:
    model, metrics = train()
MODEL_ACCURACY.set(metrics["accuracy"])


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, examples=[5.1])
    sepal_width: float = Field(..., ge=0, le=10, examples=[3.5])
    petal_length: float = Field(..., ge=0, le=10, examples=[1.4])
    petal_width: float = Field(..., ge=0, le=10, examples=[0.2])


@app.get("/health")
def health():
    REQUESTS.labels(endpoint="/health", status="200").inc()
    return {"status": "ok", "model_accuracy": metrics["accuracy"]}


@app.post("/predict")
def predict(features: IrisFeatures):
    start = time.perf_counter()
    row = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    idx = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    species = SPECIES[idx]
    PREDICTIONS.labels(species=species).inc()
    REQUESTS.labels(endpoint="/predict", status="200").inc()
    LATENCY.labels(endpoint="/predict").observe(time.perf_counter() - start)
    return {"species": species, "confidence": round(float(proba[idx]), 4)}


@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
