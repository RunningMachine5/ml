"""실제 모델 연결 전 API 흐름을 검증하는 예측기 스켈레톤."""

from typing import Protocol

from fdshield_ml.serving.schemas import PredictionRequest, PredictionResponse


class Predictor(Protocol):
    """서빙 API가 의존하는 최소 예측 인터페이스."""

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """거래 한 건의 사기 여부와 설명값을 반환한다."""


class StubPredictor:
    """전처리와 모델이 확정되기 전 사용하는 결정적 테스트 구현."""

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """API 계약 확인용 고정 결과를 반환한다."""

        return PredictionResponse(
            user_id=request.user_id,
            is_fraud=False,
            fraud_probability=0.1,
            shap={},
            model_name="stub-model",
            model_version="0",
        )
