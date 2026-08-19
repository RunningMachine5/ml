"""ML 담당자 원본 입력을 model79 숫자 행렬로 바꾸는 전처리 테스트."""

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
    preprocess_frame,
    preprocess_transaction_features,
)

RawFeaturesFactory = Callable[..., dict[str, object]]


def test_raw_and_model_contract_counts_are_fixed() -> None:
    assert len(MODEL_INPUT_COLUMNS) == len(set(MODEL_INPUT_COLUMNS)) == 51
    assert len(SERVING_INPUT_COLUMNS) == len(set(SERVING_INPUT_COLUMNS)) == 51
    assert len(TRAINING_INPUT_COLUMNS) == len(set(TRAINING_INPUT_COLUMNS)) == 53
    assert len(RAW_TRAINING_INPUT_COLUMNS) == len(set(RAW_TRAINING_INPUT_COLUMNS)) == 53
    assert len(MODEL_FEATURE_COLUMNS) == len(set(MODEL_FEATURE_COLUMNS)) == 79


def test_preprocessing_emits_exact_79_numeric_features(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    result = preprocess_transaction_features(raw_features_factory())

    assert result.shape == (1, 79)
    assert result.columns.tolist() == list(MODEL_FEATURE_COLUMNS)
    assert len(MODEL_FEATURE_COLUMNS) == len(set(MODEL_FEATURE_COLUMNS)) == 79
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


def test_preprocessing_keeps_original_category_matching_semantics(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    row = preprocess_transaction_features(
        raw_features_factory(channel=" ATM ", operating_system="iOS")
    ).iloc[0]

    assert row["channel_atm"] == 0
    assert row["operating_system_ios"] == 0


def test_train1_alias_and_metadata_produce_same_model79_vector(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    serving = raw_features_factory()
    expected = preprocess_transaction_features(serving)

    training = {
        "transaction_id": 1,
        **serving,
        "customer_name": "test-customer",
        "customer_identification_number": "synthetic-id",
        "account_account_number": "account-1",
        "error_code": "none",
        "ip_address": "127.0.0.1",
        "mac_address": "00:00:00:00:00:00",
        "location": "37.0 127.0",
        "recipient_account_number": "recipient-1",
        "first_time_ios_by_vulnerable_user": 0,
        "customer_id": 123,
        "balance_drain_ratio": 0.15,
        "is_fraud": 1,
    }
    training["account_release_suspention"] = training.pop(
        "recipient_release_suspension"
    )
    training["transaction_resumed_date"] = training.pop(
        "recipient_transaction_resumed_date"
    )
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


def test_account_type_e_is_explicit_unseen_all_zero_category(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    row = preprocess_transaction_features(
        raw_features_factory(account_account_type=" E ")
    ).iloc[0]
    encoded_columns = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column.startswith("account_account_type_")
    ]

    assert encoded_columns == [
        "account_account_type_a",
        "account_account_type_b",
        "account_account_type_c",
        "account_account_type_d",
    ]
    assert row[encoded_columns].sum() == 0


def test_optional_access_medium_is_encoded_as_all_zero(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    row = preprocess_transaction_features(
        raw_features_factory(access_medium=None)
    ).iloc[0]
    encoded_columns = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column.startswith("access_medium_")
    ]

    assert row[encoded_columns].sum() == 0
