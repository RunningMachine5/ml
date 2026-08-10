"""Git에 포함된 운영 v5 모델 번들의 로딩·예측 계약 테스트."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from fdshield_ml.common.feature_contract import MODEL_FEATURE_COLUMNS
from fdshield_ml.serving.local_predictor import (
    DEFAULT_LOCAL_MODEL_PATH,
    LocalModelPredictor,
)
from fdshield_ml.serving.model_predictor import ModelServingError
from fdshield_ml.serving.predictor import predictor_from_environment
from fdshield_ml.serving.schemas import PredictionRequest

RawFeaturesFactory = Callable[..., dict[str, object]]


def test_bundled_v5_model_predicts_with_real_shap(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    predictor = LocalModelPredictor.from_bundle(DEFAULT_LOCAL_MODEL_PATH)
    request = PredictionRequest(
        transaction_id="LOCAL-V5-001",
        features=raw_features_factory(),
    )

    first = predictor.predict(request)
    second = predictor.predict(request)

    assert first == second
    assert first.model_name == "fdshield-fraud-detector"
    assert first.model_version == "5"
    assert first.is_fraud is (first.fraud_probability >= 0.55)
    assert 0.0 <= first.fraud_probability <= 1.0
    assert list(first.shap) == list(MODEL_FEATURE_COLUMNS)
    assert len(first.shap) == 91


def test_local_model_manifest_matches_tracked_binary() -> None:
    manifest = json.loads(
        (DEFAULT_LOCAL_MODEL_PATH / "manifest.json").read_text(encoding="utf-8")
    )
    model_bytes = (DEFAULT_LOCAL_MODEL_PATH / manifest["model_file"]).read_bytes()

    actual_sha256 = hashlib.sha256(model_bytes).hexdigest()

    assert actual_sha256 == manifest["model_sha256"]
    assert manifest["model_version"] == "5"
    assert manifest["decision_threshold"] == pytest.approx(0.55)
    assert manifest["feature_count"] == 91


def test_local_model_metadata_cannot_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ML_LOCAL_MODEL_PATH", str(DEFAULT_LOCAL_MODEL_PATH))
    monkeypatch.setenv("ML_MODEL_NAME", "not-the-bundled-model")
    monkeypatch.setenv("ML_MODEL_VERSION", "999")

    predictor = LocalModelPredictor.from_environment()

    assert predictor.model_name == "fdshield-fraud-detector"
    assert predictor.model_version == "5"
    assert predictor.threshold == pytest.approx(0.55)


def test_local_model_rejects_hash_mismatch(tmp_path: Path) -> None:
    manifest = {
        "bundle_schema_version": 1,
        "model_name": "fdshield-fraud-detector",
        "model_version": "5",
        "model_format": "xgboost-ubj",
        "model_file": "model.ubj",
        "model_sha256": "0" * 64,
        "feature_count": 91,
        "decision_threshold": 0.55,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "model.ubj").write_bytes(b"not-the-tracked-model")

    with pytest.raises(ModelServingError, match="SHA-256 mismatch"):
        LocalModelPredictor.from_bundle(tmp_path)


def test_predictor_factory_defaults_to_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ML_PREDICTOR_MODE", raising=False)

    predictor = predictor_from_environment()

    assert isinstance(predictor, LocalModelPredictor)
    assert predictor.model_version == "5"


def test_predictor_factory_rejects_removed_stub_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ML_PREDICTOR_MODE", "stub")

    with pytest.raises(ValueError, match="local.*mlflow"):
        predictor_from_environment()
