import numpy as np

from ecommerce_recommender.models.metrics import precision_at_k, recall_at_k


def test_ranking_metrics_use_highest_scores() -> None:
    relevance = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.7, 0.9, 0.2])

    assert precision_at_k(relevance, scores, k=2) == 1.0
    assert recall_at_k(relevance, scores, k=2) == 1.0
