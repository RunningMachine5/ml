"""``train1.csv`` 전용 운영 학습 데이터 계약.

학습 입력은 ML 담당자가 전달한 snake_case 64열 원본 CSV 한 종류만
지원한다. 모델 입력 80열을 미리 만든 CSV나 별도 companion CSV는 더 이상
학습 계약이 아니다.
"""

from __future__ import annotations

from typing import Literal

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
TRANSACTION_ID_COLUMN = "transaction_id"
TRANSACTION_DATETIME_COLUMN = "transaction_datetime"
TRAINING_DATA_CONTRACT = "fdshield-train1-raw64-to-model80-v1"

TrainingDatasetKind = Literal["raw"]
RAW_TRAINING_COLUMNS = tuple(RAW_TRAINING_INPUT_COLUMNS)
CANONICAL_TRAINING_COLUMNS = tuple(TRAINING_INPUT_COLUMNS)
RAW_REQUIRED_COLUMNS = frozenset(RAW_TRAINING_COLUMNS)
RAW_ALLOWED_COLUMNS = RAW_REQUIRED_COLUMNS


class TrainingDatasetError(ValueError):
    """학습 CSV가 ``train1.csv`` 계약을 충족하지 않을 때 발생한다."""


def detect_training_dataset_kind(
    columns: pd.Index | list[str] | tuple[str, ...],
) -> TrainingDatasetKind:
    """실제 또는 alias 정규화 후의 train1 64열 계약만 허용한다."""

    provided = tuple(columns)
    duplicates = sorted({column for column in provided if provided.count(column) > 1})
    if duplicates:
        raise TrainingDatasetError(
            f"training data contains duplicate columns: {duplicates}"
        )

    if provided in {RAW_TRAINING_COLUMNS, CANONICAL_TRAINING_COLUMNS}:
        return "raw"

    accepted_names = set(RAW_TRAINING_COLUMNS) | set(CANONICAL_TRAINING_COLUMNS)
    missing = sorted(set(RAW_TRAINING_COLUMNS) - set(provided))
    unknown = sorted(set(provided) - accepted_names)
    if not missing and not unknown:
        raise TrainingDatasetError(
            "train1 training columns are out of order; "
            f"expected={list(RAW_TRAINING_COLUMNS)}"
        )
    raise TrainingDatasetError(
        f"invalid train1 raw64 training schema: missing={missing}, unknown={unknown}"
    )


def normalize_training_frame(source: pd.DataFrame) -> pd.DataFrame:
    """train1 alias를 정식 이름으로 바꾸고 canonical 64열 순서로 정렬한다."""

    if not isinstance(source, pd.DataFrame):
        raise TypeError("normalize_training_frame expects a pandas DataFrame")
    detect_training_dataset_kind(source.columns)
    normalized = source.rename(columns=CSV_ALIAS_COLUMNS).copy()
    if normalized.columns.duplicated().any():
        duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
        raise TrainingDatasetError(
            f"training aliases produce duplicate columns: {duplicates}"
        )
    normalized = normalized.loc[:, CANONICAL_TRAINING_COLUMNS].reset_index(drop=True)
    normalized[TRANSACTION_ID_COLUMN] = _normalize_training_transaction_ids(
        normalized[TRANSACTION_ID_COLUMN]
    )
    return normalized


def _normalize_training_transaction_ids(identifiers: pd.Series) -> pd.Series:
    """기존 train1의 ``T00000001`` ID를 정식 양의 정수 ID로 이관한다."""

    text = identifiers.astype("string").str.strip()
    empty = identifiers.isna() | text.eq("")
    if empty.any():
        rows = empty.index[empty].tolist()[:5]
        raise TrainingDatasetError(
            f"training {TRANSACTION_ID_COLUMN} must not be empty; rows={rows}"
        )

    legacy_prefixed = text.str.fullmatch(r"T\d+", na=False)
    numeric_text = text.where(~legacy_prefixed, text.str[1:])
    try:
        numeric = pd.to_numeric(numeric_text, errors="raise")
    except (TypeError, ValueError) as exc:
        raise TrainingDatasetError(
            f"training {TRANSACTION_ID_COLUMN} must contain positive integers"
        ) from exc

    values = numeric.to_numpy(dtype="float64", copy=False)
    invalid = (
        numeric.isna() | ~np.isfinite(values) | (numeric <= 0) | (numeric % 1 != 0)
    )
    if invalid.any():
        rows = invalid.index[invalid].tolist()[:5]
        raise TrainingDatasetError(
            f"training {TRANSACTION_ID_COLUMN} must contain positive integers; "
            f"rows={rows}"
        )
    return numeric.astype("int64").rename(TRANSACTION_ID_COLUMN)


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


def validate_transaction_ids(source: pd.DataFrame, *, context: str) -> pd.Series:
    """학습 행을 식별하는 transaction_id가 비어 있거나 중복되지 않게 한다."""

    if TRANSACTION_ID_COLUMN not in source:
        raise TrainingDatasetError(f"{context} data is missing {TRANSACTION_ID_COLUMN}")
    identifiers = source[TRANSACTION_ID_COLUMN]
    if not pd.api.types.is_integer_dtype(identifiers.dtype):
        raise TrainingDatasetError(
            f"{context} {TRANSACTION_ID_COLUMN} must use the normalized int contract"
        )
    if (identifiers <= 0).any():
        rows = identifiers.index[identifiers <= 0].tolist()[:5]
        raise TrainingDatasetError(
            f"{context} {TRANSACTION_ID_COLUMN} must contain positive integers; "
            f"rows={rows}"
        )
    duplicated = identifiers.duplicated(keep=False)
    if duplicated.any():
        values = identifiers.loc[duplicated].astype("string").unique().tolist()[:5]
        raise TrainingDatasetError(
            f"{context} {TRANSACTION_ID_COLUMN} must be unique; values={values}"
        )
    return identifiers.reset_index(drop=True)


__all__ = [
    "CANONICAL_TRAINING_COLUMNS",
    "LABEL_COLUMN",
    "RAW_ALLOWED_COLUMNS",
    "RAW_REQUIRED_COLUMNS",
    "RAW_TRAINING_COLUMNS",
    "TRAINING_DATA_CONTRACT",
    "TRANSACTION_DATETIME_COLUMN",
    "TRANSACTION_ID_COLUMN",
    "TrainingDatasetError",
    "TrainingDatasetKind",
    "detect_training_dataset_kind",
    "normalize_training_frame",
    "validate_binary_target",
    "validate_transaction_ids",
]
