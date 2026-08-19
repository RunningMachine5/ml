"""train1.csv 전용 raw53 학습 계약 테스트."""

from collections.abc import Callable

import pandas as pd
import pytest

from fdshield_ml.config.preprocess_config import (
    RAW_TRAINING_INPUT_COLUMNS,
    TRAINING_INPUT_COLUMNS,
)
from fdshield_ml.service.train.dataset import (
    TRAINING_DATA_CONTRACT,
    TrainingDatasetError,
    normalize_training_frame,
    validate_binary_target,
    validate_training_columns,
)
from tests.conftest import training_row_from_raw51

RawFeaturesFactory = Callable[..., dict[str, object]]


def _training_frame(
    raw_features_factory: RawFeaturesFactory,
    *,
    labels: list[object] | None = None,
) -> pd.DataFrame:
    target = labels or [0, 1, 0, 1]
    rows: list[dict[str, object]] = []
    for index, label in enumerate(target, start=1):
        row = training_row_from_raw51(
            raw_features_factory(),
            transaction_id=index,
            is_fraud=label,
        )
        rows.append(row)
    return pd.DataFrame(rows).loc[:, RAW_TRAINING_INPUT_COLUMNS]


def test_train1_alias_contract_normalizes_to_canonical_raw53_order(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    source = _training_frame(raw_features_factory)

    validate_training_columns(source.columns)
    normalized = normalize_training_frame(source)

    assert normalized.columns.tolist() == list(TRAINING_INPUT_COLUMNS)
    assert len(normalized.columns) == 53
    assert "flag_deposit_more_than_tenmillion" not in normalized
    assert "flag_deposit_more_than_ten_million" in normalized
    assert normalized["transaction_id"].tolist() == [1, 2, 3, 4]
    assert TRAINING_DATA_CONTRACT == "fdshield-train1-raw53-to-model79-v1"


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_train1_contract_rejects_schema_drift(
    raw_features_factory: RawFeaturesFactory,
    mutation: str,
) -> None:
    source = _training_frame(raw_features_factory)
    if mutation == "missing":
        source = source.drop(columns=source.columns[0])
    elif mutation == "extra":
        source["unexpected"] = 0
    with pytest.raises(TrainingDatasetError):
        validate_training_columns(source.columns)


def test_train1_contract_normalizes_input_column_order(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    source = _training_frame(raw_features_factory)
    columns = source.columns.tolist()
    columns[0], columns[1] = columns[1], columns[0]

    normalized = normalize_training_frame(source.loc[:, columns])

    assert normalized.columns.tolist() == list(TRAINING_INPUT_COLUMNS)


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


def test_training_transaction_ids_are_preserved_as_metadata(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    source = _training_frame(raw_features_factory)
    source["transaction_id"] = ["T00000001", 1, "DB-1", "DB-1"]

    normalized = normalize_training_frame(source)

    assert normalized["transaction_id"].tolist() == ["T00000001", 1, "DB-1", "DB-1"]


def test_preprocessed_model_columns_plus_label_are_not_a_training_contract() -> None:
    legacy_columns = [f"feature_{index}" for index in range(80)] + ["Is_Fraud"]

    with pytest.raises(TrainingDatasetError, match="invalid training schema"):
        validate_training_columns(legacy_columns)
