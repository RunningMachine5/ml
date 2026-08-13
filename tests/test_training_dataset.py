"""train1.csv 전용 raw64 학습 계약 테스트."""

from collections.abc import Callable

import pandas as pd
import pytest

from fdshield_ml.common.preprocess_config import (
    RAW_TRAINING_INPUT_COLUMNS,
    TRAINING_INPUT_COLUMNS,
)
from fdshield_ml.training.dataset import (
    LABEL_COLUMN,
    TRAINING_DATA_CONTRACT,
    TrainingDatasetError,
    detect_training_dataset_kind,
    normalize_training_frame,
    validate_binary_target,
    validate_transaction_ids,
)

RawFeaturesFactory = Callable[..., dict[str, object]]


def _training_frame(
    raw_features_factory: RawFeaturesFactory,
    *,
    labels: list[object] | None = None,
) -> pd.DataFrame:
    target = labels or [0, 1, 0, 1]
    rows: list[dict[str, object]] = []
    for index, label in enumerate(target, start=1):
        row = {
            "transaction_id": f"TX-{index}",
            **raw_features_factory(),
            "customer_identification_number": f"synthetic-{index}",
            "customer_id": index,
            "balance_drain_ratio": 0.1,
            LABEL_COLUMN: label,
        }
        row["flag_deposit_more_than_tenmillion"] = row.pop(
            "flag_deposit_more_than_ten_million"
        )
        rows.append(row)
    return pd.DataFrame(rows).loc[:, RAW_TRAINING_INPUT_COLUMNS]


def test_train1_alias_contract_normalizes_to_canonical_raw64_order(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    source = _training_frame(raw_features_factory)

    assert detect_training_dataset_kind(source.columns) == "raw"
    normalized = normalize_training_frame(source)

    assert normalized.columns.tolist() == list(TRAINING_INPUT_COLUMNS)
    assert len(normalized.columns) == 64
    assert "flag_deposit_more_than_tenmillion" not in normalized
    assert "flag_deposit_more_than_ten_million" in normalized
    assert TRAINING_DATA_CONTRACT == "fdshield-train1-raw64-to-model80-v1"


@pytest.mark.parametrize("mutation", ["missing", "extra", "order"])
def test_train1_contract_rejects_schema_drift(
    raw_features_factory: RawFeaturesFactory,
    mutation: str,
) -> None:
    source = _training_frame(raw_features_factory)
    if mutation == "missing":
        source = source.drop(columns=source.columns[0])
    elif mutation == "extra":
        source["unexpected"] = 0
    else:
        columns = source.columns.tolist()
        columns[0], columns[1] = columns[1], columns[0]
        source = source.loc[:, columns]

    with pytest.raises(TrainingDatasetError):
        detect_training_dataset_kind(source.columns)


@pytest.mark.parametrize("invalid_value", [0.5, "fraud", float("inf")])
def test_binary_label_rejects_non_binary_value(
    raw_features_factory: RawFeaturesFactory,
    invalid_value: object,
) -> None:
    source = normalize_training_frame(
        _training_frame(raw_features_factory, labels=[0, 1, invalid_value])
    )

    with pytest.raises(TrainingDatasetError, match="0 or 1"):
        validate_binary_target(source, context="training")


def test_binary_label_requires_both_classes(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    source = normalize_training_frame(
        _training_frame(raw_features_factory, labels=[1, 1])
    )

    with pytest.raises(TrainingDatasetError, match="both 0 and 1"):
        validate_binary_target(source, context="training")


def test_transaction_ids_must_be_present_and_unique(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    source = normalize_training_frame(_training_frame(raw_features_factory))
    source.loc[0, "transaction_id"] = ""
    with pytest.raises(TrainingDatasetError, match="must not be empty"):
        validate_transaction_ids(source, context="training")

    source = normalize_training_frame(_training_frame(raw_features_factory))
    source.loc[1, "transaction_id"] = source.loc[0, "transaction_id"]
    with pytest.raises(TrainingDatasetError, match="must be unique"):
        validate_transaction_ids(source, context="training")


def test_legacy_model80_plus_label_csv_is_not_a_training_contract() -> None:
    legacy_columns = [f"feature_{index}" for index in range(80)] + ["Is_Fraud"]

    with pytest.raises(TrainingDatasetError, match="raw64"):
        detect_training_dataset_kind(legacy_columns)
