"""운영 학습 CSV의 raw/preprocessed 계약과 시간 분할 테스트."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from fdshield_ml.common.feature_contract import MODEL_FEATURE_COLUMNS
from fdshield_ml.common.preprocessing import preprocess_frame
from fdshield_ml.training.dataset import (
    TrainingDatasetError,
    aligned_transaction_datetimes,
    detect_training_dataset_kind,
    time_split_indices,
    validate_binary_target,
    validate_preprocessed_features,
)


RawFeaturesFactory = Callable[..., dict[str, object]]


def _preprocessed_frame(
    raw_features_factory: RawFeaturesFactory,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    datetimes = [
        "2026-03-30 10:00:00",
        "2026-03-31 11:00:00",
        "2026-04-01 12:00:00",
        "2026-04-02 13:00:00",
    ]
    labels = [0, 1, 0, 1]
    raw = pd.DataFrame(
        [
            raw_features_factory(Transaction_Datetime=value)
            for value in datetimes
        ]
    )
    training = preprocess_frame(raw)
    training["Is_Fraud"] = labels
    transactions = pd.DataFrame(
        {
            "Transaction_Datetime": datetimes,
            "Is_Fraud": labels,
        }
    )
    return training, transactions


def test_exact_preprocessed_contract_is_numeric_and_ordered(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    training, _ = _preprocessed_frame(raw_features_factory)

    assert detect_training_dataset_kind(training.columns) == "preprocessed"
    features = validate_preprocessed_features(training)

    assert features.columns.tolist() == list(MODEL_FEATURE_COLUMNS)
    assert np.isfinite(features.to_numpy(dtype="float64")).all()


@pytest.mark.parametrize("mutation", ["missing", "extra", "order"])
def test_preprocessed_contract_rejects_schema_drift(
    raw_features_factory: RawFeaturesFactory,
    mutation: str,
) -> None:
    training, _ = _preprocessed_frame(raw_features_factory)
    if mutation == "missing":
        training = training.drop(columns=MODEL_FEATURE_COLUMNS[0])
    elif mutation == "extra":
        training["unexpected"] = 0
    else:
        columns = training.columns.tolist()
        columns[0], columns[1] = columns[1], columns[0]
        training = training.loc[:, columns]

    with pytest.raises(TrainingDatasetError):
        detect_training_dataset_kind(training.columns)


@pytest.mark.parametrize("invalid_value", ["not-a-number", float("inf")])
def test_preprocessed_contract_rejects_invalid_numeric_values(
    raw_features_factory: RawFeaturesFactory,
    invalid_value: object,
) -> None:
    training, _ = _preprocessed_frame(raw_features_factory)
    training[MODEL_FEATURE_COLUMNS[0]] = training[
        MODEL_FEATURE_COLUMNS[0]
    ].astype("object" if isinstance(invalid_value, str) else "float64")
    training.loc[0, MODEL_FEATURE_COLUMNS[0]] = invalid_value

    with pytest.raises(TrainingDatasetError, match="numeric|finite"):
        validate_preprocessed_features(training)


def test_binary_label_rejects_fractional_value(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    training, _ = _preprocessed_frame(raw_features_factory)
    training["Is_Fraud"] = training["Is_Fraud"].astype("float64")
    training.loc[0, "Is_Fraud"] = 0.5

    with pytest.raises(TrainingDatasetError, match="both 0 and 1"):
        validate_binary_target(training, context="training")


def test_generated_raw_metadata_columns_are_allowed(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    row = {
        "ID": "T00000001",
        **raw_features_factory(),
        "Customer_ID": "C000001",
        "Customer_personal_identifier": "홍길동",
        "Customer_identification_number": "masked",
        "Account_account_number": "account-1",
        "IP_Address": "127.0.0.1",
        "MAC_Address": "00:00:00:00:00:00",
        "Recipient_Account_Number": "recipient-1",
        "Is_Fraud": 0,
    }

    assert detect_training_dataset_kind(pd.DataFrame([row]).columns) == "raw"

    row["unknown_raw_column"] = 1
    with pytest.raises(TrainingDatasetError, match="unknown_raw_column"):
        detect_training_dataset_kind(pd.DataFrame([row]).columns)


def test_companion_alignment_and_time_split_use_row_positions(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    training, transactions = _preprocessed_frame(raw_features_factory)
    target = validate_binary_target(training, context="training")
    datetimes = aligned_transaction_datetimes(training, transactions)

    split = time_split_indices(
        datetimes,
        target,
        minimum_fraud_rows=1,
    )

    assert split.train_index.tolist() == [0, 1]
    assert split.validation_index.tolist() == [2, 3]
    assert split.train_datetime_max < split.validation_datetime_min


def test_companion_alignment_rejects_label_or_datetime_mismatch(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    training, transactions = _preprocessed_frame(raw_features_factory)
    transactions.loc[0, "Is_Fraud"] = 1

    with pytest.raises(TrainingDatasetError, match="Is_Fraud differs"):
        aligned_transaction_datetimes(training, transactions)

    training, transactions = _preprocessed_frame(raw_features_factory)
    transactions.loc[0, "Transaction_Datetime"] = "2026-03-30 15:00:00"
    with pytest.raises(TrainingDatasetError, match="transaction_hour"):
        aligned_transaction_datetimes(training, transactions)


def test_time_split_requires_minimum_fraud_rows_on_both_sides(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    training, transactions = _preprocessed_frame(raw_features_factory)
    target = validate_binary_target(training, context="training")
    datetimes = aligned_transaction_datetimes(training, transactions)

    with pytest.raises(TrainingDatasetError, match="at least 100 fraud rows"):
        time_split_indices(datetimes, target)
