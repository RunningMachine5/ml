"""ML 담당자 기준 raw 거래를 학습·추론 공용 model80 행렬로 변환한다."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fdshield_ml.config.preprocess_config import (
    CATEGORICAL_LEVELS,
    CSV_ALIAS_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
    NUMERIC_PASSTHROUGH_COLUMNS,
    OPTIONAL_ELAPSED_COLUMNS,
    REQUIRED_ELAPSED_COLUMNS,
    TRANSACTION_DATETIME_COLUMN,
)

class Preprocessor:
    """전달본과 같은 이름으로 학습·추론 공용 전처리를 제공한다."""

    def predict_preprocess(self, transaction: object) -> pd.DataFrame:
        """PredictInputDTO 한 건을 model80 행렬로 변환한다."""

        return preprocess_transaction_features(transaction.feature_values())

    def train_preprocess(self, frame: pd.DataFrame) -> pd.DataFrame:
        """raw64 학습 DataFrame을 model80 행렬로 변환한다."""

        return preprocess_frame(frame)


def preprocess_transaction_features(features: Mapping[str, object]) -> pd.DataFrame:
    """입력 경계에서 검증된 거래 한 건을 model80 Feature로 바꾼다."""

    return preprocess_frame(pd.DataFrame([dict(features)]))


def normalize_column_aliases(source_frame: pd.DataFrame) -> pd.DataFrame:
    """train1.csv의 알려진 오타를 원본 파일 변경 없이 canonical 이름으로 바꾼다."""

    return source_frame.rename(columns=CSV_ALIAS_COLUMNS).copy()


def preprocess_frame(source_frame: pd.DataFrame) -> pd.DataFrame:
    """raw60 또는 raw64 DataFrame을 순서가 고정된 model80 행렬로 바꾼다.

    train1.csv 메타데이터와 라벨은 모델 입력에서 제외한다. 비율 Feature 및
    과거 거래시각이 없는 행의 NaN은 XGBoost missing value로 보존한다.
    """

    normalized_frame = normalize_column_aliases(source_frame)
    source = normalized_frame.loc[:, MODEL_INPUT_COLUMNS].reset_index(drop=True).copy()

    result = source.loc[:, NUMERIC_PASSTHROUGH_COLUMNS].apply(pd.to_numeric)

    transaction_datetime = _datetime(source[TRANSACTION_DATETIME_COLUMN])
    customer_birth_date = _datetime(source["customer_birth_date"])
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
    result["customer_age"] = customer_age

    transaction_hour = transaction_datetime.dt.hour
    transaction_day_of_week = transaction_datetime.dt.dayofweek
    result["transaction_hour"] = transaction_hour
    result["transaction_day"] = transaction_datetime.dt.day
    result["transaction_day_of_week"] = transaction_day_of_week
    # 전달본과 동일하게 06:59:59까지 새벽으로 본다.
    result["transaction_is_dawn"] = transaction_hour.between(0, 6).astype("int8")
    result["transaction_is_weekend"] = transaction_day_of_week.ge(5).astype("int8")

    elapsed_columns = REQUIRED_ELAPSED_COLUMNS | OPTIONAL_ELAPSED_COLUMNS
    for source_column, output_column in elapsed_columns.items():
        earlier = _datetime(source[source_column])
        result[output_column] = _elapsed_days(transaction_datetime, earlier)

    seconds = _duration_seconds(source["time_difference"])
    distance = _numeric(source["distance"])
    result["seconds_since_last_transaction"] = seconds
    result["distance_since_last_transaction"] = distance
    result["distance_per_minute"] = distance.div(seconds.div(60)).where(seconds > 0)

    transaction_amount = _numeric(source["transaction_amount"])
    initial_balance = _numeric(source["account_initial_balance"])
    daily_limit = _numeric(source["account_amount_daily_limit"])
    one_month_max = _numeric(source["account_one_month_max_amount"])
    one_month_std = _numeric(source["account_one_month_std_dev"])
    dawn_month_max = _numeric(source["account_dawn_one_month_max_amount"])
    dawn_month_std = _numeric(source["account_dawn_one_month_std_dev"])
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

    categories = source.loc[:, list(CATEGORICAL_LEVELS)]
    encoded = pd.get_dummies(
        categories,
        columns=list(CATEGORICAL_LEVELS),
        dtype="float64",
    )
    return (
        pd.concat([result, encoded], axis=1)
        .reindex(columns=MODEL_FEATURE_COLUMNS, fill_value=0)
        .astype("float64")
    )


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series).astype("float64")


def _datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed")


def _elapsed_days(
    transaction_datetime: pd.Series,
    earlier_datetime: pd.Series,
) -> pd.Series:
    if earlier_datetime.isna().all():
        return pd.Series(np.nan, index=transaction_datetime.index, dtype="float64")
    return (transaction_datetime - earlier_datetime).dt.days.astype("float64")


def _duration_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series).dt.total_seconds().astype("float64")


def _positive_denominator_ratio(
    numerator: pd.Series, denominator: pd.Series
) -> pd.Series:
    return numerator.div(denominator).where(denominator > 0)


__all__ = [
    "Preprocessor",
    "normalize_column_aliases",
    "preprocess_frame",
    "preprocess_transaction_features",
]
