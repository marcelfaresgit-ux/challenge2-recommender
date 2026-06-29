import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["user_idx", "item_idx"]


def train_baseline(train: pd.DataFrame) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=120,
                    min_samples_leaf=8,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return pipeline.fit(train[FEATURES], train["target"])


def baseline_scores(model: Pipeline, frame: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(frame[FEATURES])[:, 1], index=frame.index)


def save_baseline(model: Pipeline, path: str = "data/artifacts/baseline.joblib") -> None:
    joblib.dump(model, path)
