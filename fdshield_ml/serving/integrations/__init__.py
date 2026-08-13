"""환경에 맞는 고정 모델을 로드해 PredictionService를 생성한다."""

from __future__ import annotations

import os

from fdshield_ml.serving.service.predict.predict_service import PredictService


def predict_service_from_environment() -> PredictService:
    """로컬 번들 또는 MLflow Registry의 고정 모델을 선택한다."""

    mode = os.getenv("ML_PREDICTOR_MODE", "local").strip().lower()
    if mode == "local":
        from fdshield_ml.serving.integrations.local_model import (
            load_local_predict_service_from_environment,
        )

        return load_local_predict_service_from_environment()
    if mode == "mlflow":
        from fdshield_ml.serving.integrations.mlflow_model import (
            load_mlflow_predict_service,
        )

        return load_mlflow_predict_service()
    raise ValueError("ML_PREDICTOR_MODE must be 'local' or 'mlflow'")


__all__ = ["predict_service_from_environment"]
