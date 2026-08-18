"""
[모델 학습 플로우 코드]
train1 데이터를 전처리하고 모델을 학습시키는 전체 흐름을 관리한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fdshield_ml.service.preprocessor import Preprocessor
from fdshield_ml.service.train.dataset import (
    TrainingDatasetError,
    normalize_training_frame,
    validate_binary_target,
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
    """model79 학습 행렬과 이진 라벨."""

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
    """raw64 DataFrame을 검증하고 model79 행렬로 변환한다."""

    # 데이터 계약 확인은 여기까지만 담당하고, 실제 피처 계산은 추론과 같은
    # Preprocessor에 맡긴다.
    try:
        normalized = normalize_training_frame(source)
        target = validate_binary_target(normalized, context="training")
    except TrainingDatasetError as exc:
        raise TrainingServiceError(str(exc)) from exc
    features = Preprocessor().train_preprocess(normalized)
    return PreparedTrainingData(features=features, target=target)


def train_candidate(
    source: pd.DataFrame,
    config: ModelTrainingConfig | None = None,
) -> ModelTrainingResult:
    """raw64 DataFrame 하나로 전처리와 XGBoost 후보 학습을 수행한다."""

    # 이 함수는 GCS·MLflow를 몰라야 로컬 실험과 Cloud Run Job이 함께 쓸 수 있다.
    prepared = prepare_training_data(source)
    try:
        return train_model(prepared.features, prepared.target, config)
    except ModelTrainingError as exc:
        raise TrainingServiceError(str(exc)) from exc


def ml_train_flow(
    data_path: str | Path,
    config: ModelTrainingConfig | None = None,
) -> ModelTrainingResult:
    """doo 원본과 같은 이름으로 train1 학습 전체 흐름을 제공한다."""

    # 데이터 읽기 -> 전처리 -> 모델 학습 순서로 실행한다.
    return train_candidate(load_training_frame(data_path), config)
