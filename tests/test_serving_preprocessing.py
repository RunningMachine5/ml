"""ML 담당자 원본 입력을 model80 숫자 행렬로 바꾸는 전처리 테스트."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from fdshield_ml.config.preprocess_config import (
    CATEGORICAL_LEVELS,
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
    RAW_TRAINING_INPUT_COLUMNS,
    SERVING_INPUT_COLUMNS,
    TRAINING_INPUT_COLUMNS,
)
from fdshield_ml.service.preprocessor import (
    FeaturePreprocessingError,
    preprocess_frame,
    preprocess_transaction_features,
)

RawFeaturesFactory = Callable[..., dict[str, object]]


def test_raw_and_model_contract_counts_are_fixed() -> None:
    assert len(MODEL_INPUT_COLUMNS) == len(set(MODEL_INPUT_COLUMNS)) == 59
    assert len(SERVING_INPUT_COLUMNS) == len(set(SERVING_INPUT_COLUMNS)) == 60
    assert len(TRAINING_INPUT_COLUMNS) == len(set(TRAINING_INPUT_COLUMNS)) == 64
    assert len(RAW_TRAINING_INPUT_COLUMNS) == len(set(RAW_TRAINING_INPUT_COLUMNS)) == 64
    assert len(MODEL_FEATURE_COLUMNS) == len(set(MODEL_FEATURE_COLUMNS)) == 80


def test_preprocessing_emits_exact_80_numeric_features(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    result = preprocess_transaction_features(raw_features_factory())

    assert result.shape == (1, 80)
    assert result.columns.tolist() == list(MODEL_FEATURE_COLUMNS)
    assert len(MODEL_FEATURE_COLUMNS) == len(set(MODEL_FEATURE_COLUMNS)) == 80
    assert "is_fraud" not in result
    assert not np.isinf(result.to_numpy(dtype="float64")).any()


def test_preprocessing_derives_datetime_duration_and_ratios(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    result = preprocess_transaction_features(raw_features_factory()).iloc[0]

    assert result["customer_age"] == 43
    assert result["transaction_hour"] == 3
    assert result["transaction_day"] == 12
    assert result["transaction_day_of_week"] == 6
    assert result["transaction_is_dawn"] == 1
    assert result["transaction_is_weekend"] == 1
    assert result["days_since_last_atm"] == 2
    assert np.isnan(result["days_since_last_bank_branch"])
    assert result["days_since_transaction_resumed"] == 11
    assert result["seconds_since_last_transaction"] == 93_784
    assert result["amount_to_balance_ratio"] == pytest.approx(100_000 / 8_812_467)


def test_preprocessing_creates_every_fixed_one_hot_group(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    result = preprocess_transaction_features(raw_features_factory())

    selected_levels = {
        "customer_gender": "male",
        "customer_loan_type": "c",
        "account_account_type": "a",
        "channel": "mobile",
        "operating_system": "android",
        "type_general_automatic": "general",
        "access_medium": "e",
    }
    for source_column, levels in CATEGORICAL_LEVELS.items():
        encoded_columns = [f"{source_column}_{level}" for level in levels]
        assert result[encoded_columns].sum(axis=1).item() == 1
        assert result.at[0, f"{source_column}_{selected_levels[source_column]}"] == 1


def test_preprocessing_normalizes_category_case_and_whitespace(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    row = preprocess_transaction_features(
        raw_features_factory(channel=" ATM ", operating_system="iOS")
    ).iloc[0]

    assert row["channel_atm"] == 1
    assert row["operating_system_ios"] == 1


def test_train1_alias_and_metadata_produce_same_model80_vector(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    serving = {
        "transaction_id": "TX-PARITY-1",
        **raw_features_factory(),
    }
    expected = preprocess_transaction_features(serving)

    training = {
        **serving,
        "customer_identification_number": "synthetic-id",
        "customer_id": 123,
        "balance_drain_ratio": 0.15,
        "is_fraud": 1,
    }
    training["flag_deposit_more_than_tenmillion"] = training.pop(
        "flag_deposit_more_than_ten_million"
    )
    source = pd.DataFrame([training]).loc[:, RAW_TRAINING_INPUT_COLUMNS]

    actual = preprocess_frame(source)
    pd.testing.assert_frame_equal(actual, expected)


def test_optional_operating_system_is_encoded_as_all_zero(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    row = preprocess_transaction_features(
        raw_features_factory(operating_system=None)
    ).iloc[0]
    encoded_columns = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column.startswith("operating_system_")
    ]

    assert row[encoded_columns].sum() == 0


def test_remaining_daily_limit_rejects_old_boolean_semantics(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    with pytest.raises(FeaturePreprocessingError, match="amount, not a boolean"):
        preprocess_transaction_features(
            raw_features_factory(account_remaining_amount_daily_limit_exceeded=False)
        )


def test_preprocessing_rejects_unknown_category(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    with pytest.raises(FeaturePreprocessingError, match="unknown levels"):
        preprocess_transaction_features(raw_features_factory(channel="branch"))


def test_preprocessing_rejects_negative_time_difference(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    with pytest.raises(FeaturePreprocessingError, match="greater than or equal to 0"):
        preprocess_transaction_features(
            raw_features_factory(time_difference="-1 days 23:59:59")
        )


def test_preprocessing_rejects_future_history_datetime(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    with pytest.raises(FeaturePreprocessingError, match="days_since_last_atm"):
        preprocess_transaction_features(
            raw_features_factory(
                last_atm_transaction_datetime="2025-01-13T03:04:05+09:00"
            )
        )
