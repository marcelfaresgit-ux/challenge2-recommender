import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = (scores >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def precision_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    top_indices = np.argsort(scores)[::-1][:k]
    return float(np.mean(relevance[top_indices]))


def recall_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    relevant_total = max(int(np.sum(relevance)), 1)
    top_hits = int(np.sum(relevance[np.argsort(scores)[::-1][:k]]))
    return float(top_hits / relevant_total)
