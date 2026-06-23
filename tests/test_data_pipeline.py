import pandas as pd

from ecommerce_recommender.features.prepare import add_targets, encode_column


def test_encode_column_creates_stable_indices() -> None:
    frame = pd.DataFrame({"user_id": ["u2", "u1", "u2"]})
    encoded, encoder = encode_column(frame, "user_id")

    assert encoded.tolist() == [1, 0, 1]
    assert encoder.inverse_transform([0, 1]).tolist() == ["u1", "u2"]


def test_add_targets_marks_positive_events() -> None:
    frame = pd.DataFrame({"event_type": ["view", "cart", "purchase"], "rating": [2, 3, 5]})
    prepared = add_targets(frame)

    assert prepared["target"].tolist() == [0, 1, 1]
    assert prepared["weight"].tolist() == [1.0, 2.0, 3.0]
