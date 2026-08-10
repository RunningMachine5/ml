"""환경에 맞는 실제 모델 predictor를 선택한다."""

from __future__ import annotations

import os
from typing import Protocol

from fdshield_ml.serving.schemas import PredictionRequest, PredictionResponse


class Predictor(Protocol):
    """FastAPI 애플리케이션이 의존하는 최소 추론 계약."""

    ready: bool

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """거래 한 건의 사기 여부와 설명값을 반환한다."""


def predictor_from_environment() -> Predictor:
    """로컬 번들 모델 또는 MLflow Registry 모델을 선택한다."""

    mode = os.getenv("ML_PREDICTOR_MODE", "local").strip().lower()
    if mode == "local":
        from fdshield_ml.serving.local_predictor import LocalModelPredictor

        return LocalModelPredictor.from_environment()
    if mode == "mlflow":
        from fdshield_ml.serving.mlflow_predictor import MLflowPredictor

        return MLflowPredictor.from_environment()
    raise ValueError("ML_PREDICTOR_MODE must be 'local' or 'mlflow'")
