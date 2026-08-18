"""로컬 모델 또는 MLflow 모델을 추론 서비스로 불러온다."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import xgboost as xgb

from fdshield_ml.service.predict.predict_service import (
    PredictionServiceError,
    PredictService,
)

DEFAULT_LOCAL_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "fdshield-fraud-detector-v2"
)
MANIFEST_FILE = "manifest.json"


def predict_service_from_environment() -> PredictService:
    """환경변수에 따라 로컬 모델 또는 MLflow 모델을 선택한다."""

    mode = os.getenv("ML_PREDICTOR_MODE", "local").strip().lower()
    if mode == "local":
        return load_local_predict_service_from_environment()
    if mode == "mlflow":
        return load_mlflow_predict_service()
    raise ValueError("ML_PREDICTOR_MODE must be 'local' or 'mlflow'")


def load_local_predict_service_from_environment() -> PredictService:
    """환경변수 또는 저장소 기본 경로에서 로컬 모델을 불러온다."""

    configured_path = os.getenv("ML_LOCAL_MODEL_PATH", "").strip()
    bundle_path = Path(configured_path) if configured_path else DEFAULT_LOCAL_MODEL_PATH
    return load_local_predict_service(bundle_path)


def load_local_predict_service(bundle_path: Path) -> PredictService:
    """manifest에 적힌 XGBoost 모델과 모델 정보를 불러온다."""

    manifest_path = bundle_path / MANIFEST_FILE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = xgb.XGBClassifier()
        model.load_model(bundle_path / manifest["model_file"])
    except Exception as exc:
        raise PredictionServiceError(
            f"Failed to load local model: {bundle_path}"
        ) from exc

    return PredictService(
        model=model,
        model_name=str(manifest["model_name"]),
        model_version=str(manifest["model_version"]),
    )


def load_mlflow_predict_service() -> PredictService:
    """환경변수로 지정한 MLflow Registry 모델 버전을 불러온다."""

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
    model_name = os.environ["ML_MODEL_NAME"]
    model_version = os.environ["ML_MODEL_VERSION"]

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{model_version}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        raise PredictionServiceError(
            f"Failed to load registered model: {model_uri}"
        ) from exc

    return PredictService(
        model=model,
        model_name=model_name,
        model_version=model_version,
    )


__all__ = [
    "DEFAULT_LOCAL_MODEL_PATH",
    "load_local_predict_service",
    "load_local_predict_service_from_environment",
    "load_mlflow_predict_service",
    "predict_service_from_environment",
]
