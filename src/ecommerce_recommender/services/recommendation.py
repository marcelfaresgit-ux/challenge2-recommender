from pathlib import Path

import joblib
import pandas as pd
import torch

from ecommerce_recommender.models.train import build_model
from ecommerce_recommender.utils import load_params


class RecommendationService:
    def __init__(self, artifacts_dir: Path = Path("data/artifacts")) -> None:
        self.artifacts_dir = artifacts_dir
        self.encoders = joblib.load(artifacts_dir / "encoders.joblib")
        self.train = pd.read_csv("data/processed/train.csv")
        self.items = pd.read_csv("data/raw/items.csv")
        self.model = build_model(self.train, load_params())
        state = torch.load(artifacts_dir / "recommender.pt", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def recommend(self, user_id: str, top_k: int) -> list[dict[str, object]]:
        user_idx = self._encode_user(user_id)
        item_count = len(self.encoders["item"].classes_)
        users = torch.full((item_count,), user_idx, dtype=torch.long)
        items = torch.arange(item_count, dtype=torch.long)
        scores = torch.sigmoid(self.model(users, items)).numpy()
        known_items = self._known_items(user_idx)
        ranked = self._rank_items(scores, known_items, top_k)
        return [self._format_item(item_idx, float(scores[item_idx])) for item_idx in ranked]

    def _encode_user(self, user_id: str) -> int:
        if user_id not in set(self.encoders["user"].classes_):
            raise ValueError(f"Usuario desconhecido: {user_id}")
        return int(self.encoders["user"].transform([user_id])[0])

    def _known_items(self, user_idx: int) -> set[int]:
        history = self.train.loc[self.train["user_idx"] == user_idx, "item_idx"]
        return set(history.astype(int).tolist())

    @staticmethod
    def _rank_items(scores: list[float], blocked: set[int], top_k: int) -> list[int]:
        candidates = [(idx, score) for idx, score in enumerate(scores) if idx not in blocked]
        ranked = sorted(candidates, key=lambda pair: pair[1], reverse=True)
        return [idx for idx, _ in ranked[:top_k]]

    def _format_item(self, item_idx: int, score: float) -> dict[str, object]:
        item_id = self.encoders["item"].inverse_transform([item_idx])[0]
        item = self.items.loc[self.items["item_id"] == item_id].iloc[0]
        return {
            "item_id": item_id,
            "category": item["category"],
            "price": float(item["price"]),
            "score": round(score, 4),
        }
