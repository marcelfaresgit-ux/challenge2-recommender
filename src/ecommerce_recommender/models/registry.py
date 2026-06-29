import mlflow

from ecommerce_recommender.settings import get_settings


def promote_latest_model(model_name: str = "ecommerce-neural-recommender") -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        raise RuntimeError(f"Nenhuma versao encontrada para {model_name}.")
    latest = max(versions, key=lambda version: int(version.version))
    client.transition_model_version_stage(model_name, latest.version, stage=settings.model_stage)


if __name__ == "__main__":
    promote_latest_model()
