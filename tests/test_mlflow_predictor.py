"""MLflow 모델 로딩과 추론 메타데이터 계약 테스트."""

from collections.abc import Callable

import numpy as np
import pytest

from fdshield_ml.dto.predict_input import PredictInputDTO
from fdshield_ml.infrastructure import model_loader
from fdshield_ml.infrastructure.model_loader import load_mlflow_predict_service
from fdshield_ml.service.predict.predict_service import PredictService

RawFeaturesFactory = Callable[..., dict[str, object]]


class FakeModel:
    def predict(self, features: object) -> np.ndarray:
        return np.asarray([1])

    def predict_proba(self, features: object) -> np.ndarray:
        assert features.shape == (1, 79)
        return np.asarray([[0.08, 0.92]])


def test_prediction_returns_mlflow_model_metadata(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    service = PredictService(
        model=FakeModel(),
        model_name="fdshield-fraud-detector",
        model_version="17",
    )

    result = service.predict(
        PredictInputDTO.model_validate(raw_features_factory())
    )

    assert result.predict_result == 1
    assert result.predict_proba == pytest.approx(0.92)
    assert result.model_name == "fdshield-fraud-detector"
    assert result.model_version == "17"
    assert result.shap_values == {}


def test_loader_uses_requested_registry_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    monkeypatch.setenv("ML_MODEL_NAME", "fdshield-fraud-detector")
    monkeypatch.setenv("ML_MODEL_VERSION", "17")
    calls: dict[str, str] = {}
    monkeypatch.setattr(
        model_loader.mlflow,
        "set_tracking_uri",
        lambda uri: calls.__setitem__("tracking_uri", uri),
    )
    monkeypatch.setattr(
        model_loader.mlflow.sklearn,
        "load_model",
        lambda uri: calls.__setitem__("model_uri", uri) or FakeModel(),
    )

    service = load_mlflow_predict_service()

    assert calls == {
        "tracking_uri": "https://mlflow.example.com",
        "model_uri": "models:/fdshield-fraud-detector/17",
    }
    assert service.model_name == "fdshield-fraud-detector"
    assert service.model_version == "17"
