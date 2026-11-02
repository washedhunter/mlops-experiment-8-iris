"""Unit tests for the model and the API (run by GitHub Actions on every push)."""
from fastapi.testclient import TestClient

from src.api import app
from src.train import train

client = TestClient(app)


def test_model_accuracy_above_threshold():
    _, metrics = train()
    assert metrics["accuracy"] >= 0.9


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_setosa():
    r = client.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5,
                                      "petal_length": 1.4, "petal_width": 0.2})
    assert r.status_code == 200
    assert r.json()["species"] == "setosa"


def test_predict_rejects_bad_input():
    r = client.post("/predict", json={"sepal_length": -1, "sepal_width": 3.5,
                                      "petal_length": 1.4, "petal_width": 0.2})
    assert r.status_code == 422


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "iris_predictions_total" in r.text
