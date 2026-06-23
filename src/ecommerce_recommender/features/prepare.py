from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ecommerce_recommender.utils import ensure_dir, load_params


def encode_column(frame: pd.DataFrame, column: str) -> tuple[pd.Series, LabelEncoder]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(frame[column])
    return pd.Series(encoded, name=f"{column}_idx"), encoder


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    copy = frame.copy()
    positive_event = copy["event_type"].isin(["cart", "purchase"])
    copy["target"] = (positive_event | (copy["rating"] >= 4)).astype(int)
    copy["weight"] = copy["event_type"].map({"view": 1.0, "cart": 2.0, "purchase": 3.0})
    return copy


def split_frame(
    frame: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_valid, test = train_test_split(
        frame,
        test_size=params["split"]["test_size"],
        random_state=params["seed"],
        stratify=frame["target"],
    )
    validation_size = params["split"]["validation_size"]
    train, validation = train_test_split(
        train_valid,
        test_size=validation_size,
        random_state=params["seed"],
        stratify=train_valid["target"],
    )
    return train, validation, test


def main() -> None:
    params = load_params()
    raw = pd.read_csv("data/raw/interactions.csv")
    raw["user_idx"], user_encoder = encode_column(raw, "user_id")
    raw["item_idx"], item_encoder = encode_column(raw, "item_id")
    prepared = add_targets(raw)
    train, validation, test = split_frame(prepared, params)
    output_dir = ensure_dir(Path("data/processed"))
    artifacts_dir = ensure_dir(Path("data/artifacts"))
    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    joblib.dump({"user": user_encoder, "item": item_encoder}, artifacts_dir / "encoders.joblib")


if __name__ == "__main__":
    main()
