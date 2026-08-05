"""모델 서빙 API의 임시 요청·응답 계약."""

from typing import Self

from pydantic import BaseModel, Field, model_validator

from fdshield_ml.common.feature_contract import FORBIDDEN_INFERENCE_COLUMNS


FeatureValue = str | int | float | bool | None


class PredictionRequest(BaseModel):
    """백엔드가 모델 서버에 전달하는 원본 거래 Feature 한 건."""

    transaction_id: str = Field(min_length=1)
    features: dict[str, FeatureValue] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_feature_contract(self) -> Self:
        """입력 확정 전에는 데이터 누수·민감 식별 컬럼만 차단한다."""

        provided = set(self.features)
        forbidden = sorted(provided & FORBIDDEN_INFERENCE_COLUMNS)
        if forbidden:
            raise ValueError(f"Invalid inference features: forbidden={forbidden}")
        return self


class PredictionResponse(BaseModel):
    """모델 서버가 백엔드에 반환하는 이진 분류 결과."""

    transaction_id: str
    is_fraud: bool
    fraud_probability: float = Field(ge=0.0, le=1.0)
    shap: dict[str, float] = Field(default_factory=dict)
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Cloud Run과 로컬 실행에서 사용하는 상태 응답."""

    status: str
