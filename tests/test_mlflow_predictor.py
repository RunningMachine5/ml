"""MLflow 고정 모델 버전 로딩과 실제 추론 계약 테스트."""

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from fdshield_ml.config.preprocess_config import MODEL_FEATURE_COLUMNS
from fdshield_ml.dto.predict_input import PredictInputDTO
from fdshield_ml.infrastructure import model_loader as mlflow_model
from fdshield_ml.infrastructure.model_loader import load_mlflow_predict_service
from fdshield_ml.service.predict.predict_service import (
    PredictionServiceError,
    PredictService,
)

RawFeaturesFactory = Callable[..., dict[str, object]]


class FakeProbabilityModel:
    decision_threshold_ = 0.9
    feature_names_in_ = np.asarray(MODEL_FEATURE_COLUMNS)
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features: object) -> np.ndarray:
        assert features.shape == (1, 79)
        return np.asarray([[0.08, 0.92]])


def test_mlflow_predictor_returns_registered_model_metadata(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    service = PredictService(
        model=FakeProbabilityModel(),
        model_name="fdshield-fraud-detector",
        model_version="17",
    )

    result = service.predict(
        PredictInputDTO.model_validate(
            raw_features_factory()
        )
    )

    assert result.predict_result == 1
    assert result.predict_proba == pytest.approx(0.92)
    assert result.model_name == "fdshield-fraud-detector"
    assert result.model_version == "17"
    assert result.shap_values == {}


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
        mlflow_model.mlflow,
        "set_tracking_uri",
        lambda uri: calls.__setitem__("tracking_uri", uri),
    )
    monkeypatch.setattr(
        mlflow_model.mlflow.sklearn,
        "load_model",
        lambda uri: calls.__setitem__("model_uri", uri) or FakeProbabilityModel(),
    )
    monkeypatch.setattr(
        mlflow_model,
        "MlflowClient",
        lambda: SimpleNamespace(
            get_model_version=lambda name, version: SimpleNamespace(
                tags={"decision_threshold": "0.9"}
            )
        ),
    )

    load_mlflow_predict_service()

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
        load_mlflow_predict_service()


def test_mlflow_predictor_rejects_wrong_registered_feature_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "serving")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret")
    monkeypatch.setenv("ML_MODEL_NAME", "fdshield-fraud-detector-v2")
    monkeypatch.setenv("ML_MODEL_VERSION", "1")
    wrong_model = FakeProbabilityModel()
    wrong_model.feature_names_in_ = np.asarray(MODEL_FEATURE_COLUMNS[:-1])
    monkeypatch.setattr(
        mlflow_model.mlflow.sklearn,
        "load_model",
        lambda uri: wrong_model,
    )
    monkeypatch.setattr(
        mlflow_model,
        "MlflowClient",
        lambda: SimpleNamespace(
            get_model_version=lambda name, version: SimpleNamespace(
                tags={"decision_threshold": "0.5"}
            )
        ),
    )

    with pytest.raises(PredictionServiceError, match="model79"):
        load_mlflow_predict_service()


def test_mlflow_predictor_rejects_invalid_probability_shape(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    class InvalidModel:
        decision_threshold_ = 0.5

        def predict_proba(self, features: object) -> np.ndarray:
            return np.asarray([0.5])

    service = PredictService(
        model=InvalidModel(),
        model_name="fdshield-fraud-detector",
        model_version="17",
    )

    with pytest.raises(PredictionServiceError, match="binary row"):
        service.predict(
            PredictInputDTO.model_validate(
                raw_features_factory()
            )
        )


def test_mlflow_predictor_reads_legacy_model_version_threshold_tag() -> None:
    service = PredictService(
        model=type(
            "LegacyProbabilityModel",
            (),
            {"predict_proba": lambda self, features: np.asarray([[0.4, 0.6]])},
        )(),
        model_name="fdshield-fraud-detector",
        model_version="5",
        model_version_tags={"decision_threshold": "0.55"},
    )

    assert service.threshold == pytest.approx(0.55)


def test_mlflow_predictor_rejects_model_without_threshold() -> None:
    with pytest.raises(PredictionServiceError, match="does not contain"):
        PredictService(
            model=type(
                "MissingThresholdModel",
                (),
                {"predict_proba": lambda self, features: np.asarray([[0.4, 0.6]])},
            )(),
            model_name="fdshield-fraud-detector",
            model_version="5",
        )
