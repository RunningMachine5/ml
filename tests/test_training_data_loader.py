"""로컬·GCS train1 raw64 학습 데이터 Loader 테스트."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from fdshield_ml.common.preprocess_config import RAW_TRAINING_INPUT_COLUMNS
from fdshield_ml.training.data_loader import (
    GCSObjectLocation,
    data_source_type,
    materialize_training_data,
)

RawFeaturesFactory = Callable[..., dict[str, object]]


def write_training_csv(
    path: Path,
    raw_features_factory: RawFeaturesFactory,
    *,
    labels: list[object] | None = None,
) -> None:
    target = labels or [0, 1, 0, 1]
    rows: list[dict[str, object]] = []
    for index, label in enumerate(target, start=1):
        row = {
            "transaction_id": index,
            **raw_features_factory(transaction_amount=10_000 * index),
            "customer_identification_number": f"synthetic-{index}",
            "customer_id": index,
            "balance_drain_ratio": 0.1,
            "is_fraud": label,
        }
        row["flag_deposit_more_than_tenmillion"] = row.pop(
            "flag_deposit_more_than_ten_million"
        )
        rows.append(row)
    pd.DataFrame(rows).loc[:, RAW_TRAINING_INPUT_COLUMNS].to_csv(path, index=False)


def test_data_source_type_supports_local_and_gcs() -> None:
    assert data_source_type("data/open/train1.csv") == "local"
    assert data_source_type("C:/datasets/train1.csv") == "local"
    assert data_source_type("gs://bucket/path/train1.csv") == "gcs"


def test_data_source_type_rejects_http_uri() -> None:
    with pytest.raises(ValueError, match="local path"):
        data_source_type("https://example.com/train1.csv")


def test_gcs_location_requires_bucket_and_object() -> None:
    assert GCSObjectLocation.from_uri(
        "gs://fdshield-data/training/raw/train1.csv"
    ) == GCSObjectLocation(
        bucket_name="fdshield-data",
        object_name="training/raw/train1.csv",
    )

    with pytest.raises(ValueError, match="bucket and object"):
        GCSObjectLocation.from_uri("gs://fdshield-data")


def test_local_training_data_is_used_without_copy(
    tmp_path: Path,
    raw_features_factory: RawFeaturesFactory,
) -> None:
    source = tmp_path / "train1.csv"
    write_training_csv(source, raw_features_factory)

    with materialize_training_data(str(source)) as materialized:
        assert materialized == source.resolve()
        assert materialized.read_bytes() == source.read_bytes()


def test_gcs_training_data_is_downloaded_to_temporary_file(
    tmp_path: Path,
    raw_features_factory: RawFeaturesFactory,
) -> None:
    source = tmp_path / "gcs-source.csv"
    write_training_csv(source, raw_features_factory)
    calls: dict[str, str] = {}

    class FakeBlob:
        def download_to_filename(self, filename: str) -> None:
            calls["destination"] = filename
            Path(filename).write_bytes(source.read_bytes())

    class FakeBucket:
        def blob(self, object_name: str) -> FakeBlob:
            calls["object_name"] = object_name
            return FakeBlob()

    class FakeClient:
        def bucket(self, bucket_name: str) -> FakeBucket:
            calls["bucket_name"] = bucket_name
            return FakeBucket()

    with materialize_training_data(
        "gs://fdshield-data/datasets/train1/v1/train1.csv",
        storage_client_factory=FakeClient,
    ) as materialized:
        downloaded = materialized
        assert downloaded.is_file()
        assert downloaded.read_bytes() == source.read_bytes()

    assert calls["bucket_name"] == "fdshield-data"
    assert calls["object_name"] == "datasets/train1/v1/train1.csv"
    assert not downloaded.exists()
