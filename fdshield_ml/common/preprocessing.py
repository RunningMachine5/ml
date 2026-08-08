"""원본 거래 54개를 학습·추론 공용 숫자 Feature 91개로 변환한다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from fdshield_ml.common.feature_contract import (
    CATEGORICAL_LEVELS,
    FORBIDDEN_INFERENCE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
    MODEL_INPUT_COLUMN_SET,
    NUMERIC_PASSTHROUGH_COLUMNS,
    OPTIONAL_ELAPSED_COLUMNS,
    REQUIRED_ELAPSED_COLUMNS,
    TRANSACTION_DATETIME_COLUMN,
)


TIME_DIFFERENCE_COLUMN = "Time Difference"
LOCATION_COLUMN = "Location"
DAWN_START_HOUR = 0
DAWN_END_HOUR = 6
_LOCATION_PATTERN = (
    r"(?P<latitude>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    r"(?P<longitude>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)


class FeaturePreprocessingError(ValueError):
    """원본 Feature를 안전하게 모델 행렬로 만들 수 없을 때 발생한다."""


def preprocess_transaction_features(features: Mapping[str, object]) -> pd.DataFrame:
    """JSON 형태의 원본 거래 한 건을 91개 모델 Feature로 바꾼다."""

    return preprocess_records([features])


def preprocess_records(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """JSON 형태의 원본 거래 여러 건을 순서가 고정된 숫자 행렬로 바꾼다."""

    return preprocess_frame(pd.DataFrame([dict(record) for record in records]))


def preprocess_frame(source_frame: pd.DataFrame) -> pd.DataFrame:
    """54개 원본 Feature를 순서가 고정된 91개 숫자 Feature로 바꾼다.

    여기서는 새 인코더를 fit하지 않는다. 학습 시 확정된 범주와 컬럼 순서를
    학습·Serving 양쪽에서 그대로 적용해야 Feature 불일치를 막을 수 있다.
    """

    if not isinstance(source_frame, pd.DataFrame):
        raise TypeError("preprocess_frame expects a pandas DataFrame")
    if source_frame.empty:
        raise FeaturePreprocessingError("at least one transaction is required")
    _validate_input_columns(source_frame.columns)

    source = source_frame.loc[:, MODEL_INPUT_COLUMNS].reset_index(drop=True)
    _validate_category_values(source)
    result = pd.DataFrame(index=source.index)

    # 숫자 원본 값은 이름을 바꾸지 않고 그대로 통과시킨다.
    for column in NUMERIC_PASSTHROUGH_COLUMNS:
        try:
            numeric = pd.to_numeric(source[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise FeaturePreprocessingError(
                f"{column} contains a non-numeric value"
            ) from exc
        if numeric.isna().any() or not np.isfinite(
            numeric.to_numpy(dtype="float64", copy=False)
        ).all():
            raise FeaturePreprocessingError(f"{column} must contain finite numbers")
        result[column] = numeric

    transaction_datetime = _parse_datetime(
        source[TRANSACTION_DATETIME_COLUMN],
        TRANSACTION_DATETIME_COLUMN,
        required=True,
    )
    result["transaction_hour"] = transaction_datetime.dt.hour.astype("int8")
    result["transaction_day"] = transaction_datetime.dt.day.astype("int8")
    result["transaction_dayofweek"] = transaction_datetime.dt.dayofweek.astype("int8")
    result["transaction_is_dawn"] = (
        (transaction_datetime.dt.hour >= DAWN_START_HOUR)
        & (transaction_datetime.dt.hour < DAWN_END_HOUR)
    ).astype("int8")
    result["transaction_is_weekend"] = (
        transaction_datetime.dt.dayofweek >= 5
    ).astype("int8")

    for source_column, output_column in REQUIRED_ELAPSED_COLUMNS.items():
        earlier_datetime = _parse_datetime(
            source[source_column], source_column, required=True
        )
        elapsed_days = (transaction_datetime - earlier_datetime).dt.days.astype("int64")
        if (elapsed_days < 0).any():
            raise FeaturePreprocessingError(f"{output_column} must be >= 0")
        result[output_column] = elapsed_days

    for source_column, (days_column, flag_column) in OPTIONAL_ELAPSED_COLUMNS.items():
        earlier_datetime = _parse_datetime(
            source[source_column], source_column, required=False
        )
        has_history = earlier_datetime.notna()
        elapsed_days = (transaction_datetime - earlier_datetime).dt.days
        if (elapsed_days.notna() & elapsed_days.lt(0)).any():
            raise FeaturePreprocessingError(f"{days_column} must be -1 or >= 0")
        result[days_column] = elapsed_days.fillna(-1).astype("int64")
        result[flag_column] = has_history.astype("int8")

    result["seconds_since_prev_transaction"] = _parse_time_difference(
        source[TIME_DIFFERENCE_COLUMN]
    )
    latitude, longitude = _parse_location(source[LOCATION_COLUMN])
    result["location_latitude"] = latitude
    result["location_longitude"] = longitude

    # 모든 레벨을 항상 생성하므로 이번 Batch에 없는 범주도 0 컬럼으로 유지된다.
    for column, levels in CATEGORICAL_LEVELS.items():
        for level in levels:
            result[f"{column}_{level}"] = source[column].eq(level).astype("int8")

    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in result]
    extras = [column for column in result if column not in MODEL_FEATURE_COLUMNS]
    if missing or extras:
        raise FeaturePreprocessingError(
            f"preprocessed schema mismatch: missing={missing}, unknown={extras}"
        )
    result = result.loc[:, MODEL_FEATURE_COLUMNS]
    if not np.isfinite(result.to_numpy(dtype="float64", copy=False)).all():
        raise FeaturePreprocessingError("preprocessed features must be finite numbers")
    return result


def _validate_input_columns(columns: Sequence[str]) -> None:
    provided = set(columns)
    forbidden = sorted(provided & FORBIDDEN_INFERENCE_COLUMNS)
    missing = sorted(MODEL_INPUT_COLUMN_SET - provided)
    unknown = sorted(provided - MODEL_INPUT_COLUMN_SET - FORBIDDEN_INFERENCE_COLUMNS)
    if forbidden or missing or unknown:
        raise FeaturePreprocessingError(
            "invalid inference feature contract: "
            f"forbidden={forbidden}, missing={missing}, unknown={unknown}"
        )


def _parse_datetime(
    series: pd.Series,
    column: str,
    *,
    required: bool,
) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise FeaturePreprocessingError(
            f"{column} contains an invalid datetime value"
        ) from exc
    if required and parsed.isna().any():
        raise FeaturePreprocessingError(f"{column} must not be empty")
    return parsed


def _parse_time_difference(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise FeaturePreprocessingError(f"{TIME_DIFFERENCE_COLUMN} must not be empty")
    try:
        parsed = pd.to_timedelta(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise FeaturePreprocessingError(
            f"{TIME_DIFFERENCE_COLUMN} contains an invalid duration"
        ) from exc
    total_seconds = parsed.dt.total_seconds()
    if not np.isfinite(total_seconds).all():
        raise FeaturePreprocessingError(
            f"{TIME_DIFFERENCE_COLUMN} produced a non-finite duration"
        )
    if not np.equal(total_seconds, np.floor(total_seconds)).all():
        raise FeaturePreprocessingError(
            f"{TIME_DIFFERENCE_COLUMN} must resolve to whole seconds"
        )
    if (total_seconds < 0).any():
        raise FeaturePreprocessingError("seconds_since_prev_transaction must be >= 0")
    return total_seconds.astype("int64")


def _parse_location(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    locations = series.astype("string")
    token_counts = locations.str.split().str.len()
    extracted = locations.str.extract(_LOCATION_PATTERN)
    invalid = (
        locations.isna()
        | ~token_counts.between(4, 6)
        | extracted["latitude"].isna()
        | extracted["longitude"].isna()
    )
    if invalid.any():
        raise FeaturePreprocessingError(
            f"{LOCATION_COLUMN} must contain 2-4 place tokens followed by latitude "
            "and longitude"
        )
    latitude = extracted["latitude"].astype("float64")
    longitude = extracted["longitude"].astype("float64")
    if not latitude.between(33, 39).all():
        raise FeaturePreprocessingError("location_latitude must be in [33, 39]")
    if not longitude.between(124, 132).all():
        raise FeaturePreprocessingError("location_longitude must be in [124, 132]")
    return latitude, longitude


def _validate_category_values(source: pd.DataFrame) -> None:
    for column, levels in CATEGORICAL_LEVELS.items():
        values = source[column]
        if values.isna().any():
            raise FeaturePreprocessingError(f"{column} must not be empty")
        unknown = sorted(set(values.unique()) - set(levels), key=str)
        if unknown:
            raise FeaturePreprocessingError(
                f"{column} contains unknown levels: {unknown}"
            )


__all__ = [
    "FeaturePreprocessingError",
    "preprocess_frame",
    "preprocess_records",
    "preprocess_transaction_features",
]
