"""ML 담당자 기준 raw 거래를 학습·추론 공용 model80 행렬로 변환한다."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import timedelta

import numpy as np
import pandas as pd

from fdshield_ml.config.preprocess_config import (
    CATEGORICAL_LEVELS,
    CSV_ALIAS_COLUMNS,
    FORBIDDEN_INFERENCE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_COLUMN_SET,
    MODEL_INPUT_COLUMNS,
    NUMERIC_PASSTHROUGH_COLUMNS,
    OPTIONAL_ELAPSED_COLUMNS,
    REQUIRED_ELAPSED_COLUMNS,
    SERVING_INPUT_COLUMN_SET,
    TRAINING_METADATA_COLUMNS,
    TRANSACTION_DATETIME_COLUMN,
    TRANSACTION_ID_COLUMN,
)

TIME_DIFFERENCE_COLUMN = "time_difference"
DISTANCE_COLUMN = "distance"
OPTIONAL_CATEGORICAL_COLUMNS = frozenset({"operating_system"})
_DURATION_PATTERN = re.compile(
    r"^\s*(?:(?P<days>[+-]?\d+)\s+days?\s+)?"
    r"(?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2}(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)
_ALLOWED_FRAME_METADATA = frozenset({TRANSACTION_ID_COLUMN, *TRAINING_METADATA_COLUMNS})


class FeaturePreprocessingError(ValueError):
    """원본 Feature를 안전하게 model80 행렬로 만들 수 없을 때 발생한다."""


class Preprocessor:
    """전달본과 같은 이름으로 학습·추론 공용 전처리를 제공한다."""

    def predict_preprocess(self, transaction: object) -> pd.DataFrame:
        """PredictInputDTO 한 건을 model80 행렬로 변환한다."""

        if hasattr(transaction, "feature_values"):
            features = transaction.feature_values()
        elif isinstance(transaction, Mapping):
            features = transaction
        else:
            raise TypeError("transaction must be PredictInputDTO or a mapping")
        return preprocess_transaction_features(features)

    def train_preprocess(self, frame: pd.DataFrame) -> pd.DataFrame:
        """raw64 학습 DataFrame을 model80 행렬로 변환한다."""

        return preprocess_frame(frame)


def preprocess_transaction_features(features: Mapping[str, object]) -> pd.DataFrame:
    """추론 거래 한 건을 정확한 순서의 80개 숫자 Feature로 바꾼다.

    내부 model input 59개와 정식 flat DTO의 transaction_id 포함 60개 형식을
    지원한다.
    """

    provided = set(features)
    forbidden = sorted(provided & FORBIDDEN_INFERENCE_COLUMNS)
    valid_contract = provided in (MODEL_INPUT_COLUMN_SET, SERVING_INPUT_COLUMN_SET)
    if forbidden or not valid_contract:
        missing = sorted(MODEL_INPUT_COLUMN_SET - provided)
        unknown = sorted(
            provided
            - MODEL_INPUT_COLUMN_SET
            - {TRANSACTION_ID_COLUMN}
            - FORBIDDEN_INFERENCE_COLUMNS
        )
        raise FeaturePreprocessingError(
            "invalid inference feature contract: "
            f"forbidden={forbidden}, missing={missing}, unknown={unknown}"
        )
    return preprocess_frame(pd.DataFrame([dict(features)]))


def normalize_column_aliases(source_frame: pd.DataFrame) -> pd.DataFrame:
    """train1.csv의 알려진 오타를 원본 파일 변경 없이 canonical 이름으로 바꾼다."""

    source = source_frame.copy()
    for alias, canonical in CSV_ALIAS_COLUMNS.items():
        if alias not in source:
            continue
        if canonical in source:
            raise FeaturePreprocessingError(
                f"both alias and canonical columns are present: {alias}, {canonical}"
            )
        source = source.rename(columns={alias: canonical})
    return source


def preprocess_frame(source_frame: pd.DataFrame) -> pd.DataFrame:
    """raw60 또는 raw64 DataFrame을 순서가 고정된 model80 행렬로 바꾼다.

    train1.csv 메타데이터와 라벨은 모델 입력에서 제외한다. 비율 Feature 및
    과거 거래시각이 없는 행의 NaN은 XGBoost missing value로 보존한다.
    """

    if not isinstance(source_frame, pd.DataFrame):
        raise TypeError("preprocess_frame expects a pandas DataFrame")
    if source_frame.empty:
        raise FeaturePreprocessingError("at least one transaction is required")

    normalized_frame = normalize_column_aliases(source_frame)
    _validate_frame_columns(normalized_frame.columns)
    source = normalized_frame.loc[:, MODEL_INPUT_COLUMNS].reset_index(drop=True).copy()
    _normalize_and_validate_categories(source)

    result = pd.DataFrame(index=source.index)
    for column in NUMERIC_PASSTHROUGH_COLUMNS:
        reject_boolean = column == "account_remaining_amount_daily_limit_exceeded"
        result[column] = _required_numeric(
            source[column], column, reject_boolean=reject_boolean
        )

    transaction_datetime = _parse_datetime(
        source[TRANSACTION_DATETIME_COLUMN],
        TRANSACTION_DATETIME_COLUMN,
        required=True,
    )
    customer_birth_date = _parse_datetime(
        source["customer_birth_date"], "customer_birth_date", required=True
    )
    customer_age = (
        transaction_datetime.dt.year
        - customer_birth_date.dt.year
        - (
            (transaction_datetime.dt.month < customer_birth_date.dt.month)
            | (
                (transaction_datetime.dt.month == customer_birth_date.dt.month)
                & (transaction_datetime.dt.day < customer_birth_date.dt.day)
            )
        ).astype("int8")
    )
    if (customer_age < 0).any():
        raise FeaturePreprocessingError("customer_birth_date must not be in the future")
    result["customer_age"] = customer_age

    transaction_hour = transaction_datetime.dt.hour
    transaction_day_of_week = transaction_datetime.dt.dayofweek
    result["transaction_hour"] = transaction_hour
    result["transaction_day"] = transaction_datetime.dt.day
    result["transaction_day_of_week"] = transaction_day_of_week
    # 전달본과 동일하게 06:59:59까지 새벽으로 본다.
    result["transaction_is_dawn"] = transaction_hour.between(0, 6).astype("int8")
    result["transaction_is_weekend"] = transaction_day_of_week.ge(5).astype("int8")

    for source_column, output_column in REQUIRED_ELAPSED_COLUMNS.items():
        earlier = _parse_datetime(source[source_column], source_column, required=True)
        result[output_column] = _elapsed_days(
            transaction_datetime, earlier, output_column, required=True
        )

    for source_column, output_column in OPTIONAL_ELAPSED_COLUMNS.items():
        earlier = _parse_datetime(source[source_column], source_column, required=False)
        result[output_column] = _elapsed_days(
            transaction_datetime, earlier, output_column, required=False
        )

    seconds = _duration_seconds(source[TIME_DIFFERENCE_COLUMN])
    distance = _required_numeric(source[DISTANCE_COLUMN], DISTANCE_COLUMN)
    if (seconds < 0).any():
        raise FeaturePreprocessingError(
            "seconds_since_last_transaction must be greater than or equal to 0"
        )
    if (distance < 0).any():
        raise FeaturePreprocessingError(f"{DISTANCE_COLUMN} must be >= 0")
    result["seconds_since_last_transaction"] = seconds
    result["distance_since_last_transaction"] = distance
    result["distance_per_minute"] = distance.div(seconds.div(60)).where(seconds > 0)

    transaction_amount = _required_numeric(
        source["transaction_amount"], "transaction_amount"
    )
    initial_balance = _required_numeric(
        source["account_initial_balance"], "account_initial_balance"
    )
    daily_limit = _required_numeric(
        source["account_amount_daily_limit"], "account_amount_daily_limit"
    )
    one_month_max = _required_numeric(
        source["account_one_month_max_amount"], "account_one_month_max_amount"
    )
    one_month_std = _required_numeric(
        source["account_one_month_std_dev"], "account_one_month_std_dev"
    )
    dawn_month_max = _required_numeric(
        source["account_dawn_one_month_max_amount"],
        "account_dawn_one_month_max_amount",
    )
    dawn_month_std = _required_numeric(
        source["account_dawn_one_month_std_dev"],
        "account_dawn_one_month_std_dev",
    )
    result["transaction_amount"] = transaction_amount
    result["amount_to_balance_ratio"] = _positive_denominator_ratio(
        transaction_amount, initial_balance
    )
    result["amount_to_daily_limit_ratio"] = _positive_denominator_ratio(
        transaction_amount, daily_limit
    )
    result["amount_to_one_month_max_ratio"] = _positive_denominator_ratio(
        transaction_amount, one_month_max
    )
    result["amount_to_one_month_std_dev_ratio"] = _positive_denominator_ratio(
        transaction_amount, one_month_std
    )
    dawn_mask = result["transaction_is_dawn"].eq(1)
    result["amount_to_dawn_one_month_max_ratio"] = _positive_denominator_ratio(
        transaction_amount, dawn_month_max
    ).where(dawn_mask)
    result["amount_to_dawn_one_month_std_dev_ratio"] = _positive_denominator_ratio(
        transaction_amount, dawn_month_std
    ).where(dawn_mask)

    for column, levels in CATEGORICAL_LEVELS.items():
        for level in levels:
            result[f"{column}_{level}"] = (
                source[column].eq(level).fillna(False).astype("int8")
            )

    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in result]
    extras = [column for column in result if column not in MODEL_FEATURE_COLUMNS]
    if missing or extras:
        raise FeaturePreprocessingError(
            f"preprocessed schema mismatch: missing={missing}, unknown={extras}"
        )
    result = result.loc[:, MODEL_FEATURE_COLUMNS].astype("float64")
    values = result.to_numpy(copy=False)
    if np.isinf(values).any():
        raise FeaturePreprocessingError(
            "preprocessed features must not contain infinity"
        )
    return result


def _validate_frame_columns(columns: Sequence[str]) -> None:
    provided = list(columns)
    duplicates = sorted({column for column in provided if provided.count(column) > 1})
    if duplicates:
        raise FeaturePreprocessingError(f"duplicate input columns: {duplicates}")
    provided_set = set(provided)
    missing = sorted(MODEL_INPUT_COLUMN_SET - provided_set)
    unknown = sorted(provided_set - MODEL_INPUT_COLUMN_SET - _ALLOWED_FRAME_METADATA)
    if missing or unknown:
        raise FeaturePreprocessingError(
            f"invalid raw feature contract: missing={missing}, unknown={unknown}"
        )


def _normalize_and_validate_categories(source: pd.DataFrame) -> None:
    for column, levels in CATEGORICAL_LEVELS.items():
        raw = source[column]
        normalized = raw.astype("string").str.strip().str.lower()
        normalized = normalized.mask(normalized.eq(""), pd.NA)
        if column not in OPTIONAL_CATEGORICAL_COLUMNS and normalized.isna().any():
            raise FeaturePreprocessingError(f"{column} must not be empty")
        unknown = sorted(set(normalized.dropna().unique()) - set(levels), key=str)
        if unknown:
            raise FeaturePreprocessingError(
                f"{column} contains unknown levels: {unknown}"
            )
        source[column] = normalized


def _required_numeric(
    series: pd.Series,
    column: str,
    *,
    reject_boolean: bool = False,
) -> pd.Series:
    if (
        reject_boolean
        and series.map(lambda value: isinstance(value, (bool, np.bool_))).any()
    ):
        raise FeaturePreprocessingError(
            f"{column} must contain an amount, not a boolean"
        )
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise FeaturePreprocessingError(
            f"{column} contains a non-numeric value"
        ) from exc
    values = numeric.to_numpy(dtype="float64", copy=False)
    if numeric.isna().any() or not np.isfinite(values).all():
        raise FeaturePreprocessingError(f"{column} must contain finite numbers")
    return numeric.astype("float64")


def _parse_datetime(series: pd.Series, column: str, *, required: bool) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise FeaturePreprocessingError(
            f"{column} contains an invalid datetime value"
        ) from exc
    if required and parsed.isna().any():
        raise FeaturePreprocessingError(f"{column} must not be empty")
    try:
        # Accessing .dt here also detects mixed timezone/object results early.
        _ = parsed.dt.year
    except (AttributeError, TypeError) as exc:
        raise FeaturePreprocessingError(
            f"{column} must use one consistent timezone"
        ) from exc
    return parsed


def _elapsed_days(
    transaction_datetime: pd.Series,
    earlier_datetime: pd.Series,
    output_column: str,
    *,
    required: bool,
) -> pd.Series:
    # 값이 하나도 없는 선택 컬럼은 pandas가 timezone-naive datetime64[ns]로
    # 만든다. timezone-aware 거래시각과 뺄 필요 없이 모델 missing으로 둔다.
    if not required and earlier_datetime.isna().all():
        return pd.Series(np.nan, index=transaction_datetime.index, dtype="float64")
    try:
        elapsed = (transaction_datetime - earlier_datetime).dt.days
    except (TypeError, ValueError) as exc:
        raise FeaturePreprocessingError(
            f"{output_column} datetime timezones do not match"
        ) from exc
    if required and elapsed.isna().any():
        raise FeaturePreprocessingError(f"{output_column} must not be empty")
    if (elapsed.dropna() < 0).any():
        raise FeaturePreprocessingError(f"{output_column} must be >= 0")
    return elapsed.astype("float64")


def _duration_seconds(series: pd.Series) -> pd.Series:
    def parse(value: object) -> float:
        if value is None or (
            not isinstance(value, (str, timedelta)) and pd.isna(value)
        ):
            raise FeaturePreprocessingError(
                f"{TIME_DIFFERENCE_COLUMN} must not be empty"
            )
        if isinstance(value, (bool, np.bool_)):
            raise FeaturePreprocessingError(
                f"{TIME_DIFFERENCE_COLUMN} must be a duration or seconds"
            )
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, timedelta):
            return value.total_seconds()
        if isinstance(value, np.timedelta64):
            return float(value / np.timedelta64(1, "s"))
        if isinstance(value, (str, np.str_)):
            text = str(value).strip()
            match = _DURATION_PATTERN.fullmatch(text)
            if match is None:
                try:
                    return float(text)
                except ValueError as exc:
                    raise FeaturePreprocessingError(
                        f"{TIME_DIFFERENCE_COLUMN} contains an invalid duration"
                    ) from exc
            days = int(match.group("days") or 0)
            hours = int(match.group("hours"))
            minutes = int(match.group("minutes"))
            seconds = float(match.group("seconds"))
            if hours > 23 or minutes > 59 or seconds >= 60:
                raise FeaturePreprocessingError(
                    f"{TIME_DIFFERENCE_COLUMN} contains an invalid duration"
                )
            return days * 86_400 + hours * 3_600 + minutes * 60 + seconds
        try:
            return float(pd.to_timedelta(value).total_seconds())
        except (TypeError, ValueError, OverflowError) as exc:
            raise FeaturePreprocessingError(
                f"{TIME_DIFFERENCE_COLUMN} contains an invalid duration"
            ) from exc

    result = series.map(parse).astype("float64")
    if not np.isfinite(result.to_numpy(copy=False)).all():
        raise FeaturePreprocessingError(
            f"{TIME_DIFFERENCE_COLUMN} must produce finite seconds"
        )
    return result


def _positive_denominator_ratio(
    numerator: pd.Series, denominator: pd.Series
) -> pd.Series:
    return numerator.div(denominator).where(denominator > 0)


__all__ = [
    "FeaturePreprocessingError",
    "Preprocessor",
    "normalize_column_aliases",
    "preprocess_frame",
    "preprocess_transaction_features",
]
