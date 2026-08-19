"""
[거래 데이터 전처리 프로세스]
Backend에서 받은 raw51을 원-핫 인코딩하여 ML 추론에 사용할 model79로 만든다.
학습 데이터도 같은 변환을 사용한다.
"""

from __future__ import annotations

from collections.abc import Mapping

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
        """PredictInputDTO 한 건을 model79 행렬로 변환한다."""

        return preprocess_transaction_features(transaction.model_dump())

    def train_preprocess(self, frame: pd.DataFrame) -> pd.DataFrame:
        """raw53 학습 DataFrame을 model79 행렬로 변환한다."""

        return preprocess_frame(frame)


def preprocess_transaction_features(features: Mapping[str, object]) -> pd.DataFrame:
    """입력 경계에서 검증된 거래 한 건을 model79 Feature로 바꾼다."""

    return preprocess_frame(pd.DataFrame([dict(features)]))


def normalize_column_aliases(source_frame: pd.DataFrame) -> pd.DataFrame:
    """train1.csv의 알려진 오타를 원본 파일 변경 없이 canonical 이름으로 바꾼다."""

    # 전달받은 CSV 파일은 보존하고, 메모리에서 읽은 컬럼 이름만 정리한다.
    return source_frame.rename(columns=CSV_ALIAS_COLUMNS).copy()


def preprocess_frame(source_frame: pd.DataFrame) -> pd.DataFrame:
    """raw51 또는 raw53 DataFrame을 순서가 고정된 model79 행렬로 바꾼다.

    train1.csv 메타데이터와 라벨은 모델 입력에서 제외한다. 비율 Feature 및
    과거 거래시각이 없는 행의 NaN은 XGBoost missing value로 보존한다.
    """

    normalized_frame = normalize_column_aliases(source_frame)
    # raw53 학습 파일의 거래 ID와 라벨은 모델 피처가 아니다. 학습과 추론이
    # 같은 계산을 사용하도록 양쪽 모두 여기서 raw51만 선택한다.
    source = normalized_frame.loc[:, MODEL_INPUT_COLUMNS].reset_index(drop=True).copy()

    # 모델에 그대로 들어가는 숫자·불리언 피처를 먼저 담는다.
    result = source.loc[:, NUMERIC_PASSTHROUGH_COLUMNS].apply(pd.to_numeric)

    # 고객 생년월일과 거래일을 이용해 거래 당시 나이를 계산한다.
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

    # 거래 일시에서 시간, 일자, 요일, 새벽·주말 여부를 추출한다.
    transaction_hour = transaction_datetime.dt.hour
    transaction_day_of_week = transaction_datetime.dt.dayofweek
    result["transaction_hour"] = transaction_hour
    result["transaction_day"] = transaction_datetime.dt.day
    result["transaction_day_of_week"] = transaction_day_of_week
    # 전달본과 동일하게 06:59:59까지 새벽으로 본다.
    result["transaction_is_dawn"] = transaction_hour.between(0, 6).astype("int8")
    result["transaction_is_weekend"] = transaction_day_of_week.ge(5).astype("int8")

    # 가입·계좌 생성·최근 거래로부터 며칠이 지났는지 계산한다.
    elapsed_columns = REQUIRED_ELAPSED_COLUMNS | OPTIONAL_ELAPSED_COLUMNS
    for source_column, output_column in elapsed_columns.items():
        earlier = _datetime(source[source_column])
        result[output_column] = _elapsed_days(transaction_datetime, earlier)

    # 직전 거래와의 시간·거리 차이로 이동 속도를 계산한다.
    seconds = _duration_seconds(source["time_difference"])
    distance = _numeric(source["distance"])
    result["seconds_since_last_transaction"] = seconds
    result["distance_since_last_transaction"] = distance
    result["distance_per_minute"] = distance.div(seconds.div(60)).where(seconds > 0)

    # 거래 금액을 잔액·한도·평소 거래 금액과 비교한 비율을 만든다.
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

    # 범주형 피처를 원-핫 인코딩하고 모델이 학습한 79개 순서로 정렬한다.
    categories = source.loc[:, list(CATEGORICAL_LEVELS)]
    encoded = pd.get_dummies(
        categories,
        columns=list(CATEGORICAL_LEVELS),
        dtype="float64",
    )
    return (
        pd.concat([result, encoded], axis=1)
        # XGBoost는 열 위치도 학습 당시와 같아야 하므로 항상 model79 순서로 맞춘다.
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
    # 과거 거래시각이 전부 없으면 XGBoost가 결측값으로 처리하도록 NaN을 둔다.
    if earlier_datetime.isna().all():
        return pd.Series(float("nan"), index=transaction_datetime.index)
    return (transaction_datetime - earlier_datetime).dt.days.astype("float64")


def _duration_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series).dt.total_seconds().astype("float64")


def _positive_denominator_ratio(
    numerator: pd.Series, denominator: pd.Series
) -> pd.Series:
    return numerator.div(denominator).where(denominator > 0)
