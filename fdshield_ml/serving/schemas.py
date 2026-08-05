"""모델 서빙 API의 현재 공개 데이터 기준 요청·응답 계약."""

from typing import Self

from pydantic import BaseModel, Field, model_validator

from fdshield_ml.serving.feature_contract import (
    FORBIDDEN_INFERENCE_COLUMNS,
    MODEL_INPUT_COLUMN_SET,
)


FeatureValue = str | int | float | bool | None


class PredictionRequest(BaseModel):
    """백엔드가 모델 서버에 전달하는 원본 거래 Feature 한 건."""

    transaction_id: str = Field(min_length=1)
    features: dict[str, FeatureValue] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_feature_contract(self) -> Self:
        """현재 학습 모델과 동일한 55개 원본 컬럼인지 검사한다."""

        provided = set(self.features)
        forbidden = sorted(provided & FORBIDDEN_INFERENCE_COLUMNS)
        missing = sorted(MODEL_INPUT_COLUMN_SET - provided)
        unexpected = sorted(provided - MODEL_INPUT_COLUMN_SET)

        details = []
        if forbidden:
            details.append(f"forbidden={forbidden}")
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if details:
            raise ValueError("Invalid inference features: " + ", ".join(details))
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
