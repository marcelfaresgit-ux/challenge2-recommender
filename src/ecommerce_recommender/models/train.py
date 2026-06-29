from pathlib import Path

import mlflow
import joblib
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecommerce_recommender.models.baseline import baseline_scores, save_baseline, train_baseline
from ecommerce_recommender.models.dataset import InteractionDataset
from ecommerce_recommender.models.metrics import classification_metrics
from ecommerce_recommender.models.mlp import NeuralRecommender
from ecommerce_recommender.settings import get_settings
from ecommerce_recommender.utils import ensure_dir, load_params, set_seed


def build_model(train: pd.DataFrame, params: dict) -> NeuralRecommender:
    config = params["model"]
    encoders = joblib.load("data/artifacts/encoders.joblib")
    return NeuralRecommender(
        num_users=len(encoders["user"].classes_),
        num_items=len(encoders["item"].classes_),
        embedding_dim=config["embedding_dim"],
        hidden_units=config["hidden_units"],
        dropout=config["dropout"],
    )


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    for users, items, targets in loader:
        optimizer.zero_grad()
        loss = criterion(model(users, items), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(sum(losses) / len(losses))


@torch.no_grad()
def predict(model: nn.Module, frame: pd.DataFrame) -> pd.Series:
    model.eval()
    users = torch.tensor(frame["user_idx"].to_numpy(), dtype=torch.long)
    items = torch.tensor(frame["item_idx"].to_numpy(), dtype=torch.long)
    scores = torch.sigmoid(model(users, items)).numpy()
    return pd.Series(scores, index=frame.index)


def fit_neural(train: pd.DataFrame, validation: pd.DataFrame, params: dict) -> NeuralRecommender:
    model = build_model(train, params)
    config = params["model"]
    loader = DataLoader(InteractionDataset(train), batch_size=config["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_auc = 0.0
    stale_epochs = 0
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, loader, optimizer)
        metrics = classification_metrics(validation["target"].to_numpy(), predict(model, validation))
        mlflow.log_metrics({"train_loss": train_loss, "validation_roc_auc": metrics["roc_auc"]}, step=epoch)
        best_auc, stale_epochs, best_state = update_state(
            metrics["roc_auc"],
            best_auc,
            stale_epochs,
            best_state,
            model,
        )
        if stale_epochs >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model


def update_state(
    score: float,
    best_score: float,
    stale_epochs: int,
    best_state: dict[str, torch.Tensor],
    model: nn.Module,
) -> tuple[float, int, dict[str, torch.Tensor]]:
    if score <= best_score:
        return best_score, stale_epochs + 1, best_state
    return score, 0, {key: value.detach().clone() for key, value in model.state_dict().items()}


def main() -> None:
    params = load_params()
    settings = get_settings()
    set_seed(params["seed"])
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    train = pd.read_csv("data/processed/train.csv")
    validation = pd.read_csv("data/processed/validation.csv")
    ensure_dir(Path("data/artifacts"))
    with mlflow.start_run(run_name="train-recommender"):
        mlflow.log_params(params["model"])
        baseline = train_baseline(train)
        save_baseline(baseline)
        model = fit_neural(train, validation, params)
        neural_metrics = classification_metrics(validation["target"].to_numpy(), predict(model, validation))
        baseline_metrics = classification_metrics(
            validation["target"].to_numpy(),
            baseline_scores(baseline, validation).to_numpy(),
        )
        mlflow.log_metrics({f"neural_{key}": value for key, value in neural_metrics.items()})
        mlflow.log_metrics({f"baseline_{key}": value for key, value in baseline_metrics.items()})
        torch.save(model.state_dict(), "data/artifacts/recommender.pt")


if __name__ == "__main__":
    main()
