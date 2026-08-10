"""MLflow Registry의 고정 모델 버전을 사용하는 실제 추론기."""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import numpy as np
import xgboost as xgb
from mlflow.tracking import MlflowClient

from fdshield_ml.common.decision_threshold import (
    DecisionThresholdError,
    resolve_model_decision_threshold,
)
from fdshield_ml.common.preprocessing import preprocess_transaction_features
from fdshield_ml.serving.schemas import PredictionRequest, PredictionResponse


class ModelServingError(RuntimeError):
    """모델 로딩 또는 예측 결과가 운영 계약을 만족하지 않을 때 발생한다."""


class MLflowPredictor:
    """정확한 Registry 모델 버전을 시작 시 한 번 로딩해 재사용한다."""

    def __init__(
        self,
        *,
        model: object,
        model_name: str,
        model_version: str,
        model_version_tags: dict[str, str] | None = None,
    ) -> None:
        if not model_name.strip() or not model_version.strip():
            raise ValueError("model_name and model_version are required")
        if not hasattr(model, "predict_proba"):
            raise TypeError("The registered model must provide predict_proba().")
        try:
            threshold = resolve_model_decision_threshold(
                model,
                model_version_tags=model_version_tags,
            )
        except DecisionThresholdError as exc:
            raise ModelServingError(str(exc)) from exc
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.threshold = threshold
        self.ready = True

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

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        features = preprocess_transaction_features(request.features)
        try:
            probabilities = np.asarray(self.model.predict_proba(features), dtype="float64")
        except Exception as exc:
            raise ModelServingError("Registered model prediction failed") from exc
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
            raise ModelServingError(
                f"predict_proba must return one finite binary row; shape={probabilities.shape}"
            )
        fraud_probability = float(probabilities[0, 1])
        return PredictionResponse(
            transaction_id=request.transaction_id,
            is_fraud=fraud_probability >= self.threshold,
            fraud_probability=fraud_probability,
            shap=self._shap_contributions(features),
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def _shap_contributions(self, features: object) -> dict[str, float]:
        """XGBoost 내장 기여도를 사용하고 미지원 모델은 빈 설명을 반환한다."""

        get_booster = getattr(self.model, "get_booster", None)
        if get_booster is None:
            return {}
        try:
            booster = get_booster()
            matrix = xgb.DMatrix(features, feature_names=list(features.columns))
            contributions = np.asarray(
                booster.predict(matrix, pred_contribs=True),
                dtype="float64",
            )
        except Exception as exc:
            raise ModelServingError("Failed to calculate XGBoost contributions") from exc
        if contributions.shape != (1, len(features.columns) + 1):
            raise ModelServingError(
                f"Unexpected contribution shape: {contributions.shape}"
            )
        return {
            column: float(value)
            for column, value in zip(
                features.columns,
                contributions[0, :-1],
                strict=True,
            )
        }
