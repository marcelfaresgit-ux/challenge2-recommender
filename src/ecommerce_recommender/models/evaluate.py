import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import torch

from ecommerce_recommender.models.baseline import baseline_scores
from ecommerce_recommender.models.metrics import classification_metrics, precision_at_k, recall_at_k
from ecommerce_recommender.models.train import build_model, predict
from ecommerce_recommender.settings import get_settings
from ecommerce_recommender.utils import load_params


def evaluate_scores(name: str, frame: pd.DataFrame, scores: pd.Series) -> dict[str, float]:
    metrics = classification_metrics(frame["target"].to_numpy(), scores.to_numpy())
    metrics["precision_at_10"] = precision_at_k(frame["target"].to_numpy(), scores.to_numpy(), 10)
    metrics["recall_at_10"] = recall_at_k(frame["target"].to_numpy(), scores.to_numpy(), 10)
    return {f"{name}_{key}": value for key, value in metrics.items()}


def main() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    params = load_params()
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    baseline = joblib.load("data/artifacts/baseline.joblib")
    model = build_model(train, params)
    model.load_state_dict(torch.load("data/artifacts/recommender.pt", weights_only=True))
    metrics = {}
    metrics.update(evaluate_scores("baseline", test, baseline_scores(baseline, test)))
    metrics.update(evaluate_scores("neural", test, predict(model, test)))
    Path("data/artifacts/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with mlflow.start_run(run_name="evaluate-recommender"):
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
