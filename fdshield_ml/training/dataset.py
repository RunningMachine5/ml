"""운영 학습 데이터의 스키마, 라벨, 시간 분할 계약을 검증한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from fdshield_ml.common.feature_contract import (
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
)


LABEL_COLUMN = "Is_Fraud"
GROUP_COLUMN = "Account_account_number"
TRANSACTION_DATETIME_COLUMN = "Transaction_Datetime"
DEFAULT_SPLIT_DATETIME = "2026-04-01 00:00:00"
DEFAULT_MIN_FRAUD_ROWS_PER_SPLIT = 100

TrainingDatasetKind = Literal["raw", "preprocessed"]

RAW_TRAINING_COLUMNS = (*MODEL_INPUT_COLUMNS, LABEL_COLUMN)
RAW_GROUPED_TRAINING_COLUMNS = (*MODEL_INPUT_COLUMNS, GROUP_COLUMN, LABEL_COLUMN)
PREPROCESSED_TRAINING_COLUMNS = (*MODEL_FEATURE_COLUMNS, LABEL_COLUMN)
RAW_METADATA_COLUMNS = frozenset(
    {
        "ID",
        "Customer_ID",
        "Customer_personal_identifier",
        "Customer_identification_number",
        GROUP_COLUMN,
        "IP_Address",
        "MAC_Address",
        "Recipient_Account_Number",
        LABEL_COLUMN,
    }
)
RAW_REQUIRED_COLUMNS = frozenset((*MODEL_INPUT_COLUMNS, LABEL_COLUMN))
RAW_ALLOWED_COLUMNS = frozenset(MODEL_INPUT_COLUMNS) | RAW_METADATA_COLUMNS


class TrainingDatasetError(ValueError):
    """학습 CSV가 지원하는 데이터 계약을 충족하지 않을 때 발생한다."""


@dataclass(frozen=True)
class TimeSplitIndices:
    """시간 경계로 분리한 학습·검증 행 인덱스와 관측 범위."""

    train_index: pd.Index
    validation_index: pd.Index
    boundary: pd.Timestamp
    train_datetime_min: pd.Timestamp
    train_datetime_max: pd.Timestamp
    validation_datetime_min: pd.Timestamp
    validation_datetime_max: pd.Timestamp
    train_fraud_count: int
    validation_fraud_count: int


def detect_training_dataset_kind(columns: pd.Index | list[str]) -> TrainingDatasetKind:
    """컬럼 이름과 순서를 모두 검사해 지원하는 학습 데이터 종류를 반환한다."""

    provided = list(columns)
    duplicates = sorted(
        {column for column in provided if provided.count(column) > 1}
    )
    if duplicates:
        raise TrainingDatasetError(
            f"training data contains duplicate columns: {duplicates}"
        )

    provided_set = set(provided)
    if provided == list(PREPROCESSED_TRAINING_COLUMNS):
        return "preprocessed"
    if provided_set == set(PREPROCESSED_TRAINING_COLUMNS):
        raise TrainingDatasetError(
            "preprocessed training columns are out of order; "
            f"expected={list(PREPROCESSED_TRAINING_COLUMNS)}"
        )

    if RAW_REQUIRED_COLUMNS <= provided_set:
        unknown = sorted(provided_set - RAW_ALLOWED_COLUMNS)
        if unknown:
            raise TrainingDatasetError(
                f"invalid raw training schema: missing=[], unknown={unknown}"
            )
        return "raw"

    preprocessed_missing = set(PREPROCESSED_TRAINING_COLUMNS) - provided_set
    preprocessed_unknown = provided_set - set(PREPROCESSED_TRAINING_COLUMNS)
    raw_missing = RAW_REQUIRED_COLUMNS - provided_set
    raw_unknown = provided_set - RAW_ALLOWED_COLUMNS
    if len(preprocessed_missing) + len(preprocessed_unknown) <= len(raw_missing) + len(
        raw_unknown
    ):
        kind = "preprocessed"
        missing = sorted(preprocessed_missing)
        unknown = sorted(preprocessed_unknown)
    else:
        kind = "raw"
        missing = sorted(raw_missing)
        unknown = sorted(raw_unknown)
    raise TrainingDatasetError(
        f"invalid {kind} training schema: missing={missing}, unknown={unknown}"
    )


def validate_binary_target(source: pd.DataFrame, *, context: str) -> pd.Series:
    """``Is_Fraud``가 결측·묵시적 형변환 없이 정확한 0/1 두 클래스인지 확인한다."""

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
    classes = set(numeric.unique().tolist())
    if classes != {0, 1}:
        raise TrainingDatasetError(
            f"{context} {LABEL_COLUMN} must contain both 0 and 1; "
            f"found={sorted(classes, key=str)}"
        )
    return numeric.astype("int8").rename("is_fraud")


def validate_preprocessed_features(source: pd.DataFrame) -> pd.DataFrame:
    """91개 모델 피처를 정해진 순서의 유한 숫자 행렬로 검증한다."""

    if detect_training_dataset_kind(source.columns) != "preprocessed":
        raise TrainingDatasetError("expected the preprocessed 91-feature contract")

    result = pd.DataFrame(index=source.index)
    invalid_numeric: list[str] = []
    invalid_finite: list[str] = []
    for column in MODEL_FEATURE_COLUMNS:
        try:
            numeric = pd.to_numeric(source[column], errors="raise")
        except (TypeError, ValueError):
            invalid_numeric.append(column)
            continue
        values = numeric.to_numpy(dtype="float64", copy=False)
        if numeric.isna().any() or not np.isfinite(values).all():
            invalid_finite.append(column)
            continue
        result[column] = numeric

    if invalid_numeric:
        raise TrainingDatasetError(
            "preprocessed features must be numeric; "
            f"invalid={invalid_numeric}"
        )
    if invalid_finite:
        raise TrainingDatasetError(
            "preprocessed features must contain finite values; "
            f"invalid={invalid_finite}"
        )
    return result.loc[:, MODEL_FEATURE_COLUMNS]


def aligned_transaction_datetimes(
    training_source: pd.DataFrame,
    transactions_source: pd.DataFrame,
) -> pd.Series:
    """전처리 데이터와 원본 거래의 행·라벨·시간 파생값 정렬을 검증한다."""

    if len(training_source) != len(transactions_source):
        raise TrainingDatasetError(
            "row count mismatch breaks positional datetime alignment: "
            f"training={len(training_source)}, transactions={len(transactions_source)}"
        )
    missing = {
        TRANSACTION_DATETIME_COLUMN,
        LABEL_COLUMN,
    } - set(transactions_source.columns)
    if missing:
        raise TrainingDatasetError(
            f"transactions data is missing required columns: {sorted(missing)}"
        )

    training_target = validate_binary_target(training_source, context="training")
    transaction_target = validate_binary_target(
        transactions_source, context="transactions"
    )
    if not np.array_equal(
        training_target.to_numpy(), transaction_target.to_numpy()
    ):
        mismatch_positions = np.flatnonzero(
            training_target.to_numpy() != transaction_target.to_numpy()
        )[:5]
        raise TrainingDatasetError(
            "positional row alignment failed: Is_Fraud differs between training "
            f"and transactions data at rows={mismatch_positions.tolist()}"
        )

    datetimes = transaction_datetimes(transactions_source, context="transactions")

    alignment_checks = {
        "transaction_hour": datetimes.dt.hour,
        "transaction_day": datetimes.dt.day,
        "transaction_dayofweek": datetimes.dt.dayofweek,
    }
    for column, expected in alignment_checks.items():
        if column not in training_source:
            raise TrainingDatasetError(
                f"preprocessed training data is missing alignment column {column}"
            )
        actual = pd.to_numeric(training_source[column], errors="coerce")
        matches = actual.eq(expected)
        if not matches.all():
            rows = matches.index[~matches].tolist()[:5]
            raise TrainingDatasetError(
                "positional datetime alignment failed: "
                f"{column} does not match {TRANSACTION_DATETIME_COLUMN}; rows={rows}"
            )
    return datetimes.reset_index(drop=True)


def transaction_datetimes(source: pd.DataFrame, *, context: str) -> pd.Series:
    """원본 거래 시각을 결측 없는 datetime Series로 검증한다."""

    if TRANSACTION_DATETIME_COLUMN not in source:
        raise TrainingDatasetError(
            f"{context} data is missing {TRANSACTION_DATETIME_COLUMN}"
        )
    try:
        datetimes = pd.to_datetime(
            source[TRANSACTION_DATETIME_COLUMN], errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise TrainingDatasetError(
            f"{TRANSACTION_DATETIME_COLUMN} contains an invalid datetime value"
        ) from exc
    if datetimes.isna().any():
        rows = datetimes.index[datetimes.isna()].tolist()[:5]
        raise TrainingDatasetError(
            f"{TRANSACTION_DATETIME_COLUMN} must not be empty; rows={rows}"
        )
    return datetimes.reset_index(drop=True)


def time_split_indices(
    datetimes: pd.Series,
    target: pd.Series,
    *,
    split_datetime: datetime | str | pd.Timestamp = DEFAULT_SPLIT_DATETIME,
    minimum_fraud_rows: int = DEFAULT_MIN_FRAUD_ROWS_PER_SPLIT,
) -> TimeSplitIndices:
    """과거를 학습, 경계 이후를 검증으로 나누고 최소 사기 건수를 보장한다."""

    if minimum_fraud_rows < 1:
        raise TrainingDatasetError("minimum_fraud_rows must be positive")
    try:
        boundary = pd.Timestamp(split_datetime)
    except (TypeError, ValueError) as exc:
        raise TrainingDatasetError("split_datetime must be a valid datetime") from exc
    if pd.isna(boundary):
        raise TrainingDatasetError("split_datetime must be a valid datetime")

    aligned_datetimes = datetimes.reset_index(drop=True)
    aligned_target = target.reset_index(drop=True)
    if len(aligned_datetimes) != len(aligned_target):
        raise TrainingDatasetError(
            "datetime and target row counts must match for time split"
        )
    try:
        train_mask = aligned_datetimes.lt(boundary)
        validation_mask = aligned_datetimes.ge(boundary)
    except TypeError as exc:
        raise TrainingDatasetError(
            "split_datetime timezone must match Transaction_Datetime"
        ) from exc
    if not train_mask.any() or not validation_mask.any():
        raise TrainingDatasetError(
            "time split must produce non-empty train and validation sides; "
            f"boundary={boundary}"
        )

    train_datetimes = aligned_datetimes.loc[train_mask]
    validation_datetimes = aligned_datetimes.loc[validation_mask]
    if not train_datetimes.max() < validation_datetimes.min():
        raise TrainingDatasetError(
            "time split is not disjoint: "
            f"train_max={train_datetimes.max()}, "
            f"validation_min={validation_datetimes.min()}"
        )

    train_fraud_count = int(aligned_target.loc[train_mask].eq(1).sum())
    validation_fraud_count = int(aligned_target.loc[validation_mask].eq(1).sum())
    train_normal_count = int(aligned_target.loc[train_mask].eq(0).sum())
    validation_normal_count = int(aligned_target.loc[validation_mask].eq(0).sum())
    if train_normal_count == 0 or validation_normal_count == 0:
        raise TrainingDatasetError(
            "each time split must contain normal rows; "
            f"training={train_normal_count}, validation={validation_normal_count}"
        )
    if (
        train_fraud_count < minimum_fraud_rows
        or validation_fraud_count < minimum_fraud_rows
    ):
        raise TrainingDatasetError(
            f"each time split needs at least {minimum_fraud_rows} fraud rows; "
            f"training={train_fraud_count}, validation={validation_fraud_count}. "
            "Check split_datetime."
        )

    return TimeSplitIndices(
        train_index=pd.Index(np.flatnonzero(train_mask.to_numpy())),
        validation_index=pd.Index(np.flatnonzero(validation_mask.to_numpy())),
        boundary=boundary,
        train_datetime_min=train_datetimes.min(),
        train_datetime_max=train_datetimes.max(),
        validation_datetime_min=validation_datetimes.min(),
        validation_datetime_max=validation_datetimes.max(),
        train_fraud_count=train_fraud_count,
        validation_fraud_count=validation_fraud_count,
    )


__all__ = [
    "DEFAULT_MIN_FRAUD_ROWS_PER_SPLIT",
    "DEFAULT_SPLIT_DATETIME",
    "GROUP_COLUMN",
    "LABEL_COLUMN",
    "PREPROCESSED_TRAINING_COLUMNS",
    "RAW_GROUPED_TRAINING_COLUMNS",
    "RAW_ALLOWED_COLUMNS",
    "RAW_METADATA_COLUMNS",
    "RAW_REQUIRED_COLUMNS",
    "RAW_TRAINING_COLUMNS",
    "TRANSACTION_DATETIME_COLUMN",
    "TimeSplitIndices",
    "TrainingDatasetError",
    "TrainingDatasetKind",
    "aligned_transaction_datetimes",
    "detect_training_dataset_kind",
    "time_split_indices",
    "transaction_datetimes",
    "validate_binary_target",
    "validate_preprocessed_features",
]
