"""Train a RandomForest classifier on the Iris dataset and save the model + metrics.

Usage:  python -m src.train
Outputs: model/model.joblib, model/metrics.json
"""
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES = ["setosa", "versicolor", "virginica"]


def train(random_state: int = 42):
    """Train the model and return (model, metrics)."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return model, metrics


def main():
    model, metrics = train()
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics: {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
