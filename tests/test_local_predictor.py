"""저장소에 포함된 로컬 model79 로딩·예측 테스트."""

from collections.abc import Callable

import pytest

from fdshield_ml.dto.predict_input import PredictInputDTO
from fdshield_ml.infrastructure.model_loader import (
    DEFAULT_LOCAL_MODEL_PATH,
    load_local_predict_service,
    load_local_predict_service_from_environment,
    predict_service_from_environment,
)
from fdshield_ml.service.predict.predict_service import PredictService

RawFeaturesFactory = Callable[..., dict[str, object]]


def test_bundled_model_predicts_with_shap(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    service = load_local_predict_service(DEFAULT_LOCAL_MODEL_PATH)
    request = PredictInputDTO.model_validate(raw_features_factory())

    first = service.predict(request)
    second = service.predict(request)

    assert first == second
    assert first.model_name == "fdshield-fraud-detector-v2"
    assert first.model_version == "1"
    assert first.predict_result in (0, 1)
    assert 0.0 <= first.predict_proba <= 1.0
    assert len(first.shap_values) == 55


def test_bundled_model_accepts_optional_values(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    service = load_local_predict_service(DEFAULT_LOCAL_MODEL_PATH)
    request = PredictInputDTO.model_validate(
        raw_features_factory(
            account_account_type="e",
            access_medium=None,
        )
    )

    result = service.predict(request)

    assert result.model_name == "fdshield-fraud-detector-v2"
    assert result.model_version == "1"
    assert result.predict_result in (0, 1)
    assert 0.0 <= result.predict_proba <= 1.0
    assert len(result.shap_values) == 55


def test_local_model_metadata_comes_from_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ML_LOCAL_MODEL_PATH", str(DEFAULT_LOCAL_MODEL_PATH))
    monkeypatch.setenv("ML_MODEL_NAME", "not-the-bundled-model")
    monkeypatch.setenv("ML_MODEL_VERSION", "999")

    service = load_local_predict_service_from_environment()

    assert service.model_name == "fdshield-fraud-detector-v2"
    assert service.model_version == "1"


def test_predictor_defaults_to_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ML_PREDICTOR_MODE", raising=False)

    service = predict_service_from_environment()

    assert isinstance(service, PredictService)
    assert service.model_version == "1"


def test_predictor_rejects_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ML_PREDICTOR_MODE", "stub")

    with pytest.raises(ValueError, match="local.*mlflow"):
        predict_service_from_environment()
