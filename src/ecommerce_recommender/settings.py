import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "ecommerce-recommender"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "ecommerce-recommendation"
    model_stage: str = "Production"
    top_k: int = 10
    random_seed: int = 42
    project_root: Path = Path(__file__).resolve().parents[2]

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "data" / "artifacts"


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "ecommerce-recommender"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "ecommerce-recommendation"),
        model_stage=os.getenv("MODEL_STAGE", "Production"),
        top_k=int(os.getenv("TOP_K", "10")),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
    )
