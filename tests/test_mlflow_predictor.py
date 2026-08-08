"""MLflow 고정 모델 버전 로딩과 실제 추론 계약 테스트."""

from collections.abc import Callable

import numpy as np
import pytest

from fdshield_ml.serving import mlflow_predictor
from fdshield_ml.serving.mlflow_predictor import MLflowPredictor, ModelServingError
from fdshield_ml.serving.schemas import PredictionRequest


RawFeaturesFactory = Callable[..., dict[str, object]]


class FakeProbabilityModel:
    def predict_proba(self, features: object) -> np.ndarray:
        assert getattr(features, "shape") == (1, 91)
        return np.asarray([[0.08, 0.92]])


def test_mlflow_predictor_returns_registered_model_metadata(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    predictor = MLflowPredictor(
        model=FakeProbabilityModel(),
        model_name="fdshield-fraud-detector",
        model_version="17",
        threshold=0.5,
    )

    result = predictor.predict(
        PredictionRequest(
            transaction_id="TX-17",
            features=raw_features_factory(),
        )
    )

    assert result.is_fraud is True
    assert result.fraud_probability == pytest.approx(0.92)
    assert result.model_name == "fdshield-fraud-detector"
    assert result.model_version == "17"
    assert result.shap == {}


def test_mlflow_predictor_loads_exact_numeric_registry_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "serving")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret")
    monkeypatch.setenv("ML_MODEL_NAME", "fdshield-fraud-detector")
    monkeypatch.setenv("ML_MODEL_VERSION", "17")
    calls: dict[str, str] = {}
    monkeypatch.setattr(
        mlflow_predictor.mlflow,
        "set_tracking_uri",
        lambda uri: calls.__setitem__("tracking_uri", uri),
    )
    monkeypatch.setattr(
        mlflow_predictor.mlflow.sklearn,
        "load_model",
        lambda uri: calls.__setitem__("model_uri", uri) or FakeProbabilityModel(),
    )

    predictor = MLflowPredictor.from_environment()

    assert predictor.ready is True
    assert calls == {
        "tracking_uri": "https://mlflow.example.com",
        "model_uri": "models:/fdshield-fraud-detector/17",
    }


def test_mlflow_predictor_rejects_mutable_latest_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "serving")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret")
    monkeypatch.setenv("ML_MODEL_NAME", "fdshield-fraud-detector")
    monkeypatch.setenv("ML_MODEL_VERSION", "latest")

    with pytest.raises(ValueError, match="exact numeric"):
        MLflowPredictor.from_environment()


def test_mlflow_predictor_rejects_invalid_probability_shape(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    class InvalidModel:
        def predict_proba(self, features: object) -> np.ndarray:
            return np.asarray([0.5])

    predictor = MLflowPredictor(
        model=InvalidModel(),
        model_name="fdshield-fraud-detector",
        model_version="17",
    )

    with pytest.raises(ModelServingError, match="binary row"):
        predictor.predict(
            PredictionRequest(
                transaction_id="TX-BAD",
                features=raw_features_factory(),
            )
        )
