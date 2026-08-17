"""MLflow나 GCS 없이 train1.csv로 로컬 모델 번들을 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from fdshield_ml.config.preprocess_config import MODEL_FEATURE_COLUMNS
from fdshield_ml.infrastructure.model_loader import load_local_predict_service
from fdshield_ml.service.predict.predict_service import PredictionServiceError
from fdshield_ml.service.train.model_training import ModelTrainingConfig
from fdshield_ml.service.train.train_service import TrainingServiceError, ml_train_flow

DEFAULT_DATA_PATH = Path("data/open/train1.csv")
DEFAULT_OUTPUT_PATH = Path("models/local-training-output")
FEATURE_CONTRACT_VERSION = "raw51-model79-v1"


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _prepare_output_directory(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {resolved}. Choose a new --output-dir."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def train_local_bundle(
    *,
    data_path: Path,
    output_dir: Path,
    model_name: str,
    model_version: str,
    config: ModelTrainingConfig | None = None,
) -> dict[str, object]:
    """공용 core로 학습하고 Serving이 즉시 읽을 수 있는 번들을 저장한다."""

    normalized_name = model_name.strip()
    if not normalized_name:
        raise ValueError("model_name is required")
    if not model_version.isdigit() or int(model_version) <= 0:
        raise ValueError("model_version must be a positive integer")

    bundle_path = _prepare_output_directory(output_dir)
    result = ml_train_flow(data_path, config)
    model_path = bundle_path / "model.json"
    result.model.save_model(str(model_path))

    metrics_document = {
        "metrics": result.metrics,
        "train_rows": result.train_rows,
        "validation_rows": result.validation_rows,
        "iteration_range": result.iteration_range,
    }
    (bundle_path / "metrics.json").write_text(
        json.dumps(metrics_document, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "bundle_schema_version": 1,
        "model_name": normalized_name,
        "model_version": model_version,
        "model_format": "xgboost-json",
        "model_file": model_path.name,
        "model_sha256": _sha256(model_path),
        "decision_threshold": result.decision_threshold,
        "decision_threshold_source": "trained_model",
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
    }
    (bundle_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # 로컬 학습 결과가 실제 Serving loader 계약을 통과하는지 저장 직후 확인한다.
    load_local_predict_service(bundle_path)
    return {
        "bundle_path": str(bundle_path),
        "model_name": normalized_name,
        "model_version": model_version,
        **metrics_document,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a local model79 bundle from train1.csv without MLflow.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-name", default="fdshield-fraud-detector-v2")
    parser.add_argument("--model-version", default="1")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = train_local_bundle(
            data_path=args.data,
            output_dir=args.output_dir,
            model_name=args.model_name,
            model_version=args.model_version,
        )
    except (
        FileExistsError,
        OSError,
        PredictionServiceError,
        TrainingServiceError,
        ValueError,
    ) as exc:
        print(str(exc), file=stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_OUTPUT_PATH",
    "build_parser",
    "main",
    "train_local_bundle",
]
