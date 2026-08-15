"""MLflow 없이도 사용할 수 있는 train1 전처리·학습 흐름."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fdshield_ml.config.preprocess_config import MODEL_FEATURE_COLUMNS
from fdshield_ml.service.preprocessor import Preprocessor
from fdshield_ml.service.train.dataset import (
    TrainingDatasetError,
    normalize_training_frame,
    validate_binary_target,
    validate_transaction_ids,
)
from fdshield_ml.service.train.model_training import (
    ModelTrainingConfig,
    ModelTrainingError,
    ModelTrainingResult,
    train_model,
)


class TrainingServiceError(RuntimeError):
    """학습 CSV 검증·전처리·모델 학습이 완료되지 못했을 때 발생한다."""


@dataclass(frozen=True)
class PreparedTrainingData:
    """검증을 마친 model80 학습 행렬과 이진 라벨."""

    features: pd.DataFrame
    target: pd.Series


def load_training_frame(data_path: str | Path) -> pd.DataFrame:
    """로컬에 준비된 train1 CSV를 읽는다."""

    try:
        source = pd.read_csv(Path(data_path), low_memory=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise TrainingServiceError(f"Failed to read training CSV: {data_path}") from exc
    if len(source) < 10:
        raise TrainingServiceError("Training data must contain at least 10 rows.")
    return source


def prepare_training_data(source: pd.DataFrame) -> PreparedTrainingData:
    """raw64 DataFrame을 검증하고 정확한 model80 행렬로 변환한다."""

    try:
        normalized = normalize_training_frame(source)
        target = validate_binary_target(normalized, context="training")
        validate_transaction_ids(normalized, context="training")
    except TrainingDatasetError as exc:
        raise TrainingServiceError(str(exc)) from exc
    features = Preprocessor().train_preprocess(normalized)

    if features.columns.tolist() != list(MODEL_FEATURE_COLUMNS):
        raise TrainingServiceError(
            "Preprocessed training schema is not the model80 contract."
        )
    return PreparedTrainingData(features=features, target=target)


def train_candidate(
    source: pd.DataFrame,
    config: ModelTrainingConfig | None = None,
) -> ModelTrainingResult:
    """raw64 DataFrame 하나로 전처리와 XGBoost 후보 학습을 수행한다."""

    prepared = prepare_training_data(source)
    try:
        return train_model(prepared.features, prepared.target, config)
    except ModelTrainingError as exc:
        raise TrainingServiceError(str(exc)) from exc


def ml_train_flow(
    data_path: str | Path,
    config: ModelTrainingConfig | None = None,
) -> ModelTrainingResult:
    """전달본과 같은 이름으로 로컬 train1 학습 전체 흐름을 제공한다."""

    return train_candidate(load_training_frame(data_path), config)


__all__ = [
    "PreparedTrainingData",
    "TrainingServiceError",
    "load_training_frame",
    "ml_train_flow",
    "prepare_training_data",
    "train_candidate",
]
