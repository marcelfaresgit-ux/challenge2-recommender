from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., examples=["u_00042"])
    top_k: int = Field(10, ge=1, le=50)


class RecommendedItem(BaseModel):
    item_id: str
    category: str
    price: float
    score: float


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendedItem]
