"""``train1.csv`` 전용 운영 학습 데이터 계약.

학습 입력은 ML 담당자가 전달한 snake_case 64열 원본 CSV 한 종류만
지원한다. 모델 입력 79열을 미리 만든 CSV나 별도 companion CSV는 더 이상
학습 계약이 아니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fdshield_ml.config.preprocess_config import (
    CSV_ALIAS_COLUMNS,
    RAW_TRAINING_INPUT_COLUMNS,
    TRAINING_INPUT_COLUMNS,
)
from fdshield_ml.config.preprocess_config import (
    LABEL_COLUMN as CONTRACT_LABEL_COLUMN,
)

LABEL_COLUMN = CONTRACT_LABEL_COLUMN
TRAINING_DATA_CONTRACT = "fdshield-train1-raw64-to-model79-v1"

RAW_TRAINING_COLUMNS = tuple(RAW_TRAINING_INPUT_COLUMNS)
CANONICAL_TRAINING_COLUMNS = tuple(TRAINING_INPUT_COLUMNS)
ACCEPTED_COLUMNS = set(RAW_TRAINING_COLUMNS) | set(CANONICAL_TRAINING_COLUMNS)


class TrainingDatasetError(ValueError):
    """학습 CSV가 학습 데이터 계약을 충족하지 않을 때 발생한다."""


def validate_training_columns(
    columns: pd.Index | list[str] | tuple[str, ...],
) -> None:
    """학습 데이터 53열 계약을 검증하고 누락 및 미확인 컬럼을 차단한다."""

    provided = tuple(columns)
    duplicates = sorted({column for column in provided if provided.count(column) > 1})
    if duplicates:
        raise TrainingDatasetError(
            f"training data contains duplicate columns: {duplicates}"
        )

    missing = sorted(
        canonical
        for canonical, raw in zip(CANONICAL_TRAINING_COLUMNS, RAW_TRAINING_COLUMNS, strict=True)
        if canonical not in provided and raw not in provided
    )
    unknown = sorted(set(provided) - ACCEPTED_COLUMNS)
    if not missing and not unknown:
        return
    raise TrainingDatasetError(
        f"invalid training schema: missing={missing}, unknown={unknown}"
    )


def normalize_training_frame(source: pd.DataFrame) -> pd.DataFrame:
    """train alias를 정식 이름으로 바꾸고 canonical 53열 순서로 정렬한다.

    ``transaction_id``는 모델 입력이 아닌 행 식별용 메타데이터이므로
    원본 CSV 값을 변환하지 않고 그대로 보존한다.
    """

    if not isinstance(source, pd.DataFrame):
        raise TypeError("normalize_training_frame expects a pandas DataFrame")
    validate_training_columns(source.columns)
    normalized = source.rename(columns=CSV_ALIAS_COLUMNS).copy()
    if normalized.columns.duplicated().any():
        duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
        raise TrainingDatasetError(
            f"training aliases produce duplicate columns: {duplicates}"
        )
    return normalized.loc[:, CANONICAL_TRAINING_COLUMNS].reset_index(drop=True)


def validate_binary_target(source: pd.DataFrame, *, context: str) -> pd.Series:
    """``is_fraud``가 결측이나 묵시적 반올림 없는 정확한 0/1인지 검사한다."""

    if LABEL_COLUMN not in source:
        raise TrainingDatasetError(f"{context} data is missing {LABEL_COLUMN}")
    try:
        numeric = pd.to_numeric(source[LABEL_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise TrainingDatasetError(
            f"{context} {LABEL_COLUMN} must contain only 0 or 1"
        ) from exc

    values = numeric.to_numpy(dtype="float64", copy=False)
    if numeric.isna().any() or not np.isfinite(values).all():
        raise TrainingDatasetError(
            f"{context} {LABEL_COLUMN} must contain finite 0 or 1 values"
        )
    invalid = sorted(set(numeric.unique().tolist()) - {0, 1}, key=str)
    if invalid:
        raise TrainingDatasetError(
            f"{context} {LABEL_COLUMN} must contain only 0 or 1; invalid={invalid}"
        )
    classes = set(numeric.unique().tolist())
    if classes != {0, 1}:
        raise TrainingDatasetError(
            f"{context} {LABEL_COLUMN} must contain both 0 and 1; "
            f"found={sorted(classes, key=str)}"
        )
    return numeric.astype("int8").rename(LABEL_COLUMN)
