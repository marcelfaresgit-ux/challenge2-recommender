from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_recommender.utils import ensure_dir, load_params, set_seed

CATEGORIES = ["eletronicos", "casa", "moda", "beleza", "esporte", "livros"]
EVENT_WEIGHTS = {"view": 1.0, "cart": 3.0, "purchase": 5.0}


def build_users(total: int, rng: np.random.Generator) -> pd.DataFrame:
    segments = ["economico", "recorrente", "premium", "ocasional"]
    return pd.DataFrame(
        {
            "user_id": [f"u_{idx:05d}" for idx in range(total)],
            "segment": rng.choice(segments, total, p=[0.35, 0.3, 0.15, 0.2]),
            "age": rng.integers(18, 66, total),
        }
    )


def build_items(total: int, rng: np.random.Generator) -> pd.DataFrame:
    prices = np.round(rng.lognormal(mean=4.2, sigma=0.55, size=total), 2)
    return pd.DataFrame(
        {
            "item_id": [f"i_{idx:05d}" for idx in range(total)],
            "category": rng.choice(CATEGORIES, total),
            "price": prices,
            "popularity": rng.beta(2, 5, total),
        }
    )


def score_interaction(user: pd.Series, item: pd.Series, rng: np.random.Generator) -> float:
    segment_boost = {"economico": -0.25, "recorrente": 0.2, "premium": 0.45, "ocasional": -0.1}
    category_noise = rng.normal(0, 0.45)
    price_penalty = -0.35 if user.segment == "economico" and item.price > 120 else 0.0
    return item.popularity + segment_boost[user.segment] + price_penalty + category_noise


def build_interactions(
    users: pd.DataFrame,
    items: pd.DataFrame,
    total: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for idx in range(total):
        user = users.sample(1, random_state=idx).iloc[0]
        item = items.sample(1, weights="popularity", random_state=idx).iloc[0]
        score = score_interaction(user, item, rng)
        event = choose_event(score, rng)
        rows.append(make_row(idx, user, item, event, score, rng))
    return pd.DataFrame(rows)


def choose_event(score: float, rng: np.random.Generator) -> str:
    purchase_p = float(np.clip(0.08 + score * 0.12, 0.02, 0.75))
    cart_p = float(np.clip(0.2 + score * 0.1, purchase_p + 0.02, 0.85))
    return rng.choice(["view", "cart", "purchase"], p=[1 - cart_p, cart_p - purchase_p, purchase_p])


def make_row(
    idx: int,
    user: pd.Series,
    item: pd.Series,
    event: str,
    score: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    rating = int(np.clip(round(2.8 + score + EVENT_WEIGHTS[event] / 2), 1, 5))
    return {
        "interaction_id": idx,
        "user_id": user.user_id,
        "item_id": item.item_id,
        "event_type": event,
        "rating": rating,
        "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=int(rng.integers(0, 4380))),
    }


def main() -> None:
    params = load_params()
    set_seed(params["seed"])
    rng = np.random.default_rng(params["seed"])
    output_dir = ensure_dir(Path("data/raw"))
    users = build_users(params["data"]["users"], rng)
    items = build_items(params["data"]["items"], rng)
    interactions = build_interactions(users, items, params["data"]["interactions"], rng)
    users.to_csv(output_dir / "users.csv", index=False)
    items.to_csv(output_dir / "items.csv", index=False)
    interactions.to_csv(output_dir / "interactions.csv", index=False)


if __name__ == "__main__":
    main()
