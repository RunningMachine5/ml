"""Git에 포함된 고정 XGBoost 모델 번들 loader."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from fdshield_ml.common.decision_threshold import validate_decision_threshold
from fdshield_ml.common.preprocess_config import MODEL_FEATURE_COLUMNS
from fdshield_ml.serving.service.predict.predict_service import (
    PredictionServiceError,
    PredictService,
)

DEFAULT_LOCAL_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "fdshield-fraud-detector-v2"
)
MANIFEST_FILE = "manifest.json"


def load_local_predict_service_from_environment() -> PredictService:
    """환경변수 또는 저장소 기본 경로의 로컬 모델을 로드한다."""

    configured_path = os.getenv("ML_LOCAL_MODEL_PATH", "").strip()
    bundle_path = (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else DEFAULT_LOCAL_MODEL_PATH.resolve()
    )
    return load_local_predict_service(bundle_path)


def load_local_predict_service(bundle_path: Path) -> PredictService:
    """manifest와 해시를 검증한 뒤 PredictionService를 생성한다."""

    manifest_path = bundle_path / MANIFEST_FILE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionServiceError(
            f"Failed to read local model manifest: {manifest_path}"
        ) from exc
    if manifest.get("bundle_schema_version") != 1:
        raise PredictionServiceError("unsupported local model bundle_schema_version")
    if manifest.get("feature_count") != len(MODEL_FEATURE_COLUMNS):
        raise PredictionServiceError(
            "local model manifest feature_count does not match the active contract"
        )

    model_name = _required_string(manifest, "model_name")
    model_version = _required_string(manifest, "model_version")
    if not model_version.isdigit() or int(model_version) <= 0:
        raise PredictionServiceError("local model_version must be a positive integer")
    model_format = _required_string(manifest, "model_format")
    if model_format not in {"xgboost-ubj", "xgboost-json"}:
        raise PredictionServiceError(
            "local model_format must be xgboost-ubj or xgboost-json"
        )

    model_file = _required_string(manifest, "model_file")
    model_path = (bundle_path / model_file).resolve()
    try:
        model_path.relative_to(bundle_path.resolve())
    except ValueError as exc:
        raise PredictionServiceError(
            "local model_file must stay inside its bundle"
        ) from exc
    expected_hash = _required_string(manifest, "model_sha256").lower()
    actual_hash = _sha256(model_path)
    if actual_hash != expected_hash:
        raise PredictionServiceError(
            "local model SHA-256 mismatch; restore the tracked model bundle"
        )

    try:
        threshold = validate_decision_threshold(
            manifest["decision_threshold"],
            source="local model manifest decision threshold",
        )
    except (KeyError, ValueError) as exc:
        raise PredictionServiceError("invalid local model decision_threshold") from exc

    model = xgb.XGBClassifier()
    try:
        model.load_model(model_path)
    except Exception as exc:
        raise PredictionServiceError(
            f"Failed to load local model: {model_path}"
        ) from exc
    _validate_xgboost_contract(model)
    return PredictService(
        model=model,
        model_name=model_name,
        model_version=model_version,
        model_version_tags={"decision_threshold": repr(threshold)},
    )


__all__ = [
    "DEFAULT_LOCAL_MODEL_PATH",
    "load_local_predict_service",
    "load_local_predict_service_from_environment",
]


def _required_string(manifest: dict[str, Any], name: str) -> str:
    value = manifest.get(name)
    if not isinstance(value, str) or not value.strip():
        raise PredictionServiceError(f"local model manifest requires {name}")
    return value.strip()


def _sha256(path: Path) -> str:
    try:
        with path.open("rb") as model_file:
            digest = hashlib.file_digest(model_file, "sha256")
    except OSError as exc:
        raise PredictionServiceError(
            f"Failed to read local model file: {path}"
        ) from exc
    return digest.hexdigest()


def _validate_xgboost_contract(model: xgb.XGBClassifier) -> None:
    try:
        booster = model.get_booster()
        feature_names = booster.feature_names
        classes = np.asarray(model.classes_)
    except Exception as exc:
        raise PredictionServiceError(
            "local model does not expose the XGBoost contract"
        ) from exc
    if booster.num_features() != len(MODEL_FEATURE_COLUMNS):
        raise PredictionServiceError(
            "local model feature count does not match the active contract"
        )
    if feature_names != list(MODEL_FEATURE_COLUMNS):
        raise PredictionServiceError(
            "local model feature names or order do not match the contract"
        )
    if classes.shape != (2,) or not np.array_equal(classes, np.asarray([0, 1])):
        raise PredictionServiceError("local model must be a binary 0/1 classifier")
