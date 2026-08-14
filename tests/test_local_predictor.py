"""Git에 포함된 운영 model80 번들의 로딩·예측 계약 테스트."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from fdshield_ml.dto.predict_input import (
    CUSTOMER_FEATURE_DEFAULTS,
    DERIVED_FEATURE_DEFAULTS,
    PredictInputDTO,
)
from fdshield_ml.infrastructure.model_loader import (
    DEFAULT_LOCAL_MODEL_PATH,
    load_local_predict_service,
    load_local_predict_service_from_environment,
    predict_service_from_environment,
)
from fdshield_ml.service.predict.predict_service import (
    PredictionServiceError,
    PredictService,
)

RawFeaturesFactory = Callable[..., dict[str, object]]


def test_bundled_model80_predicts_with_real_shap(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    service = load_local_predict_service(DEFAULT_LOCAL_MODEL_PATH)
    request = PredictInputDTO.model_validate(
        {
            "transaction_id": 1001,
            **raw_features_factory(),
        }
    )

    first = service.predict(request)
    second = service.predict(request)

    assert first == second
    assert first.model_name == "fdshield-fraud-detector-v2"
    assert first.model_version == "1"
    assert first.predict_result == int(first.predict_proba >= 0.5)
    assert 0.0 <= first.predict_proba <= 1.0
    assert len(first.shap_values) == 56


def test_bundled_model80_predicts_with_pr118_compatible_missing_values(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    service = load_local_predict_service(DEFAULT_LOCAL_MODEL_PATH)
    request = PredictInputDTO.model_validate(
        {
            "transaction_id": 1002,
            **raw_features_factory(
                account_account_type="e",
                account_initial_balance=None,
                account_balance=None,
                account_remaining_amount_daily_limit_exceeded=None,
                access_medium=None,
            ),
        }
    )

    result = service.predict(request)

    assert result.model_name == "fdshield-fraud-detector-v2"
    assert result.model_version == "1"
    assert result.predict_result == int(result.predict_proba >= 0.5)
    assert 0.0 <= result.predict_proba <= 1.0
    assert len(result.shap_values) == 56


def test_bundled_model80_predicts_without_customer_and_derived_features(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    """임시 기본값이 테스트 모델이 아닌 실제 XGBoost에서도 동작한다."""

    raw_features = raw_features_factory()
    for field_name in (*CUSTOMER_FEATURE_DEFAULTS, *DERIVED_FEATURE_DEFAULTS):
        raw_features.pop(field_name)
    raw_features.pop("recipient_account_number")

    request = PredictInputDTO.model_validate({"transaction_id": 1003, **raw_features})
    result = load_local_predict_service(DEFAULT_LOCAL_MODEL_PATH).predict(request)

    assert result.model_name == "fdshield-fraud-detector-v2"
    assert result.model_version == "1"
    assert 0.0 <= result.predict_proba <= 1.0
    assert len(result.shap_values) == 56


def test_local_model_manifest_matches_tracked_binary() -> None:
    manifest = json.loads(
        (DEFAULT_LOCAL_MODEL_PATH / "manifest.json").read_text(encoding="utf-8")
    )
    model_bytes = (DEFAULT_LOCAL_MODEL_PATH / manifest["model_file"]).read_bytes()

    actual_sha256 = hashlib.sha256(model_bytes).hexdigest()

    assert actual_sha256 == manifest["model_sha256"]
    assert manifest["model_version"] == "1"
    assert manifest["decision_threshold"] == pytest.approx(0.5)
    assert manifest["feature_count"] == 80
    assert manifest["feature_contract_version"] == "raw60-model80-v1"


def test_local_model_metadata_cannot_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ML_LOCAL_MODEL_PATH", str(DEFAULT_LOCAL_MODEL_PATH))
    monkeypatch.setenv("ML_MODEL_NAME", "not-the-bundled-model")
    monkeypatch.setenv("ML_MODEL_VERSION", "999")

    service = load_local_predict_service_from_environment()

    assert service.model_name == "fdshield-fraud-detector-v2"
    assert service.model_version == "1"
    assert service.threshold == pytest.approx(0.5)


def test_local_model_rejects_hash_mismatch(tmp_path: Path) -> None:
    manifest = {
        "bundle_schema_version": 1,
        "model_name": "fdshield-fraud-detector-v2",
        "model_version": "1",
        "model_format": "xgboost-json",
        "model_file": "model.json",
        "model_sha256": "0" * 64,
        "feature_count": 80,
        "decision_threshold": 0.5,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "model.json").write_bytes(b"not-the-tracked-model")

    with pytest.raises(PredictionServiceError, match="SHA-256 mismatch"):
        load_local_predict_service(tmp_path)


def test_predictor_factory_defaults_to_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ML_PREDICTOR_MODE", raising=False)

    service = predict_service_from_environment()

    assert isinstance(service, PredictService)
    assert service.model_version == "1"


def test_predictor_factory_rejects_removed_stub_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ML_PREDICTOR_MODE", "stub")

    with pytest.raises(ValueError, match="local.*mlflow"):
        predict_service_from_environment()
