"""로컬·GCS 학습 데이터 Loader 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fdshield_ml.training.data_loader import (
    GCSObjectLocation,
    TrainingDataError,
    data_source_type,
    inspect_training_csv,
    materialize_training_data,
)


def write_training_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "Account_account_number": ["account-1", "account-2", "account-3"],
            "Transaction_Amount": [10_000, 50_000, 90_000],
            "Fraud_Type": ["m", "a", "l"],
        }
    ).to_csv(path, index=False)


def test_data_source_type_supports_local_gcs_and_stub() -> None:
    assert data_source_type("data/open/train.csv") == "local"
    assert data_source_type("C:/datasets/train.csv") == "local"
    assert data_source_type("gs://bucket/path/train.csv") == "gcs"
    assert data_source_type("stub://local-data") == "stub"


def test_data_source_type_rejects_http_uri() -> None:
    with pytest.raises(ValueError, match="local path"):
        data_source_type("https://example.com/train.csv")


def test_gcs_location_requires_bucket_and_object() -> None:
    assert GCSObjectLocation.from_uri(
        "gs://fdshield-data/training/raw/train.csv"
    ) == GCSObjectLocation(
        bucket_name="fdshield-data",
        object_name="training/raw/train.csv",
    )

    with pytest.raises(ValueError, match="bucket and object"):
        GCSObjectLocation.from_uri("gs://fdshield-data")


def test_local_training_data_is_used_without_copy(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    write_training_csv(source)

    with materialize_training_data(str(source)) as materialized:
        assert materialized == source.resolve()
        assert materialized.read_bytes() == source.read_bytes()


def test_gcs_training_data_is_downloaded_to_temporary_file(tmp_path: Path) -> None:
    source = tmp_path / "gcs-source.csv"
    write_training_csv(source)
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
        "gs://fdshield-data/datasets/open/v1/train.csv",
        storage_client_factory=FakeClient,
    ) as materialized:
        downloaded = materialized
        assert downloaded.is_file()
        assert downloaded.read_bytes() == source.read_bytes()

    assert calls["bucket_name"] == "fdshield-data"
    assert calls["object_name"] == "datasets/open/v1/train.csv"
    assert not downloaded.exists()


def test_training_csv_summary_contains_only_counts(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    write_training_csv(source)

    summary = inspect_training_csv(source, chunk_size=2)

    assert summary.row_count == 3
    assert summary.column_count == 3
    assert summary.normal_count == 1
    assert summary.fraud_count == 2
    assert summary.file_size_bytes == source.stat().st_size


def test_training_csv_rejects_missing_required_column(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    pd.DataFrame({"Fraud_Type": ["m"]}).to_csv(source, index=False)

    with pytest.raises(TrainingDataError, match="Account_account_number"):
        inspect_training_csv(source)


def test_training_csv_rejects_invalid_label(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Account_account_number": ["account-1"],
            "Fraud_Type": ["unknown"],
        }
    ).to_csv(source, index=False)

    with pytest.raises(TrainingDataError, match="Invalid training labels"):
        inspect_training_csv(source)
