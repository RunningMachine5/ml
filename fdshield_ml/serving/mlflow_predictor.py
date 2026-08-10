"""MLflow Registry의 고정 모델 버전을 사용하는 실제 추론기."""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from fdshield_ml.serving.model_predictor import ModelPredictor, ModelServingError


class MLflowPredictor(ModelPredictor):
    """정확한 Registry 모델 버전을 시작 시 한 번 로딩해 재사용한다."""

    @classmethod
    def from_environment(cls) -> MLflowPredictor:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip().rstrip("/")
        username = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
        password = os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()
        model_name = os.getenv("ML_MODEL_NAME", "").strip()
        model_version = os.getenv("ML_MODEL_VERSION", "").strip()
        if not tracking_uri.startswith(("http://", "https://")):
            raise ValueError("MLFLOW_TRACKING_URI must be an HTTP(S) URL")
        if not username or not password:
            raise ValueError("MLflow username and password are required")
        if not model_name:
            raise ValueError("ML_MODEL_NAME is required")
        if not model_version.isdigit():
            raise ValueError("ML_MODEL_VERSION must be an exact numeric Registry version")

        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{model_name}/{model_version}"
        try:
            model = mlflow.sklearn.load_model(model_uri)
            version = MlflowClient().get_model_version(model_name, model_version)
        except Exception as exc:
            raise ModelServingError(f"Failed to load registered model: {model_uri}") from exc
        return cls(
            model=model,
            model_name=model_name,
            model_version=model_version,
            model_version_tags=getattr(version, "tags", None),
        )
