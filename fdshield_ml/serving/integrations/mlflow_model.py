"""MLflow Registry의 고정 모델 버전 loader."""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient

from fdshield_ml.common.preprocess_config import MODEL_FEATURE_COLUMNS
from fdshield_ml.serving.service.predict.predict_service import (
    PredictionServiceError,
    PredictService,
)


def load_mlflow_predict_service() -> PredictService:
    """환경변수로 지정한 정확한 Registry 모델 버전을 로드한다."""

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
        raise PredictionServiceError(
            f"Failed to load registered model: {model_uri}"
        ) from exc
    _validate_registered_model_contract(model)
    return PredictService(
        model=model,
        model_name=model_name,
        model_version=model_version,
        model_version_tags=getattr(version, "tags", None),
    )


__all__ = ["load_mlflow_predict_service"]


def _validate_registered_model_contract(model: object) -> None:
    """Ready 전에 Registry 모델의 model80 순서와 이진 class 계약을 확정한다."""

    names = getattr(model, "feature_names_in_", None)
    if names is None:
        try:
            names = model.get_booster().feature_names  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            names = None
    if names is None or tuple(str(name) for name in names) != MODEL_FEATURE_COLUMNS:
        raise PredictionServiceError(
            "Registered model feature names or order do not match model80"
        )
    classes = getattr(model, "classes_", None)
    if classes is None or not np.array_equal(np.asarray(classes), np.asarray([0, 1])):
        raise PredictionServiceError(
            "Registered model must expose binary classes [0, 1]"
        )
