from functools import lru_cache

from fastapi import FastAPI, HTTPException

from ecommerce_recommender.api.schemas import RecommendationRequest, RecommendationResponse
from ecommerce_recommender.services.recommendation import RecommendationService
from ecommerce_recommender.settings import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")


@lru_cache
def get_service() -> RecommendationService:
    return RecommendationService(settings.artifacts_dir)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(payload: RecommendationRequest) -> RecommendationResponse:
    try:
        items = get_service().recommend(payload.user_id, payload.top_k)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Modelo ainda nao treinado.") from error
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return RecommendationResponse(user_id=payload.user_id, recommendations=items)
