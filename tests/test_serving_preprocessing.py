"""Backend 원본 입력을 모델 숫자 행렬로 바꾸는 전처리 테스트."""

from collections.abc import Callable

import numpy as np
import pytest

from fdshield_ml.common.feature_contract import (
    CATEGORICAL_LEVELS,
    MODEL_FEATURE_COLUMNS,
)
from fdshield_ml.common.preprocessing import (
    FeaturePreprocessingError,
    preprocess_transaction_features,
)


RawFeaturesFactory = Callable[..., dict[str, object]]


def test_preprocessing_emits_exact_91_numeric_features(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    result = preprocess_transaction_features(raw_features_factory())

    assert result.shape == (1, 91)
    assert result.columns.tolist() == list(MODEL_FEATURE_COLUMNS)
    assert len(MODEL_FEATURE_COLUMNS) == len(set(MODEL_FEATURE_COLUMNS)) == 91
    assert "Is_Fraud" not in result
    assert np.isfinite(result.to_numpy(dtype="float64")).all()


def test_preprocessing_derives_datetime_duration_and_location(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    result = preprocess_transaction_features(raw_features_factory()).iloc[0]

    assert result["transaction_hour"] == 3
    assert result["transaction_day"] == 12
    assert result["transaction_dayofweek"] == 6
    assert result["transaction_is_dawn"] == 1
    assert result["transaction_is_weekend"] == 1
    assert result["days_since_last_atm"] == 2
    assert result["has_atm_history"] == 1
    assert result["days_since_last_branch"] == -1
    assert result["has_branch_history"] == 0
    assert result["days_since_transaction_resumed"] == 11
    assert result["has_resumed_history"] == 1
    assert result["seconds_since_prev_transaction"] == 93_784
    assert result["location_latitude"] == pytest.approx(35.4164)
    assert result["location_longitude"] == pytest.approx(127.3904)


def test_preprocessing_creates_every_fixed_one_hot_level(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    result = preprocess_transaction_features(raw_features_factory())

    selected_levels = {
        "Customer_Gender": "male",
        "Customer_loan_type": "c",
        "Account_account_type": "a",
        "Channel": "mobile",
        "Operating_System": "Android",
        "Error_Code": "a",
        "Type_General_Automatic": "general",
        "Access_Medium": "e",
    }
    for source_column, levels in CATEGORICAL_LEVELS.items():
        encoded_columns = [f"{source_column}_{level}" for level in levels]
        assert result[encoded_columns].sum(axis=1).item() == 1
        assert result.at[0, f"{source_column}_{selected_levels[source_column]}"] == 1


def test_preprocessing_rejects_unknown_category(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    with pytest.raises(FeaturePreprocessingError, match="unknown level"):
        preprocess_transaction_features(raw_features_factory(Channel="branch"))


def test_preprocessing_rejects_negative_time_difference(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    with pytest.raises(FeaturePreprocessingError, match=">= 0"):
        preprocess_transaction_features(
            raw_features_factory(**{"Time Difference": "-1 days 23:59:59"})
        )


def test_preprocessing_rejects_future_history_datetime(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    with pytest.raises(FeaturePreprocessingError, match="days_since_last_atm"):
        preprocess_transaction_features(
            raw_features_factory(
                Last_atm_transaction_datetime="2025-01-13 03:04:05"
            )
        )
