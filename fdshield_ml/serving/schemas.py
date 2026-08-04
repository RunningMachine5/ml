"""모델 서빙 API의 임시 요청·응답 계약."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TransactionFeatures(BaseModel):
    """현재 Backend 거래 DTO에서 모델 추론에 사용하는 필드."""

    # 최종 피처가 확정되기 전까지 새 필드를 허용해 Skeleton 간 연동을 막지 않는다.
    model_config = ConfigDict(extra="allow")

    transaction_time: datetime
    amount: int = Field(ge=0)
    user_amount_std_dev: float = Field(gt=0.0)
    payment_method: str
    merchant_category: str


class PredictionRequest(BaseModel):
    """백엔드가 모델 서버에 전달하는 거래 한 건."""

    user_id: str
    features: TransactionFeatures


class PredictionResponse(BaseModel):
    """모델 서버가 백엔드에 반환하는 이진 분류 결과."""

    user_id: str
    is_fraud: bool
    fraud_probability: float = Field(ge=0.0, le=1.0)
    shap: dict[str, float] = Field(default_factory=dict)
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Cloud Run과 로컬 실행에서 사용하는 상태 응답."""

    status: str
