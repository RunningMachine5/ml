"""ML 추론 결과와 상태 응답 DTO."""

from pydantic import BaseModel, Field


class PredictResultDTO(BaseModel):
    """ML 예측 결과와 운영 추적용 모델 메타데이터."""

    transaction_id: str
    predict_result: int = Field(ge=0, le=1)
    predict_proba: float = Field(ge=0.0, le=1.0)
    shap_values: dict[str, float] = Field(default_factory=dict)
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Cloud Run과 로컬 실행에서 사용하는 상태 응답."""

    status: str


__all__ = ["HealthResponse", "PredictResultDTO"]
