"""로컬 파일과 GCS 객체를 동일한 학습 데이터 경로로 다루는 Loader."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import pandas as pd
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from google.cloud import storage

from fdshield_ml.common.feature_contract import MODEL_INPUT_COLUMNS
from fdshield_ml.common.preprocessing import (
    FeaturePreprocessingError,
    preprocess_frame,
)
from fdshield_ml.common.features import (
    GROUP_COLUMN,
    TARGET_COLUMN,
    binary_target,
)
from fdshield_ml.training.dataset import (
    DEFAULT_MIN_FRAUD_ROWS_PER_SPLIT,
    DEFAULT_SPLIT_DATETIME,
    LABEL_COLUMN,
    TRANSACTION_DATETIME_COLUMN,
    TrainingDatasetError,
    aligned_transaction_datetimes,
    detect_training_dataset_kind,
    time_split_indices,
    transaction_datetimes,
    validate_binary_target,
    validate_preprocessed_features,
)


DataSourceType = Literal["gcs", "local", "stub"]
StorageClientFactory = Callable[[], Any]


class TrainingDataError(RuntimeError):
    """학습 데이터 경로, 다운로드 또는 CSV 검증에 실패한 경우."""


@dataclass(frozen=True)
class GCSObjectLocation:
    """검증된 GCS Bucket과 객체 이름."""

    bucket_name: str
    object_name: str

    @classmethod
    def from_uri(cls, uri: str) -> "GCSObjectLocation":
        parsed = urlparse(uri)
        bucket_name = parsed.netloc.strip()
        object_name = unquote(parsed.path.lstrip("/"))

        if parsed.scheme.lower() != "gs" or not bucket_name or not object_name:
            raise ValueError(
                "TRAINING_DATA_URI must include a GCS bucket and object path."
            )
        if parsed.query or parsed.fragment:
            raise ValueError("TRAINING_DATA_URI must not include query or fragment.")

        return cls(bucket_name=bucket_name, object_name=object_name)


@dataclass(frozen=True)
class TrainingDataSummary:
    """개인정보나 원본 행을 포함하지 않는 학습 데이터 검증 결과."""

    row_count: int
    column_count: int
    normal_count: int
    fraud_count: int
    file_size_bytes: int


def data_source_type(data_uri: str) -> DataSourceType:
    """환경변수 값이 Stub, GCS 또는 로컬 경로인지 구분한다."""

    normalized = data_uri.strip()
    lowered = normalized.lower()

    if lowered.startswith("stub://"):
        return "stub"
    if lowered.startswith("gs://"):
        GCSObjectLocation.from_uri(normalized)
        return "gcs"
    if "://" in normalized:
        raise ValueError(
            "TRAINING_DATA_URI must be a local path, gs:// URI, or stub:// URI."
        )
    return "local"


@contextmanager
def materialize_training_data(
    data_uri: str,
    *,
    storage_client_factory: StorageClientFactory = storage.Client,
) -> Iterator[Path]:
    """로컬 경로는 그대로 사용하고 GCS 객체만 임시 파일로 다운로드한다."""

    source_type = data_source_type(data_uri)
    if source_type == "stub":
        raise TrainingDataError("Stub data URI does not contain a training file.")

    if source_type == "local":
        local_path = Path(data_uri).expanduser().resolve()
        if not local_path.is_file():
            raise TrainingDataError(f"Training CSV not found: {local_path}")
        yield local_path
        return

    location = GCSObjectLocation.from_uri(data_uri)
    file_name = Path(location.object_name).name or "training-data.csv"

    with TemporaryDirectory(prefix="fdshield-training-") as temporary_directory:
        destination = Path(temporary_directory) / file_name
        try:
            client = storage_client_factory()
            blob = client.bucket(location.bucket_name).blob(location.object_name)
            blob.download_to_filename(str(destination))
        except (GoogleAPIError, GoogleAuthError, OSError) as error:
            raise TrainingDataError(
                f"Failed to download training data from GCS: {data_uri}"
            ) from error

        if not destination.is_file() or destination.stat().st_size == 0:
            raise TrainingDataError(f"Downloaded training data is empty: {data_uri}")

        yield destination


def inspect_training_csv(
    data_path: Path,
    *,
    chunk_size: int = 50_000,
    transactions_path: Path | None = None,
    split_datetime: str = DEFAULT_SPLIT_DATETIME,
    minimum_fraud_rows_per_split: int = DEFAULT_MIN_FRAUD_ROWS_PER_SPLIT,
) -> TrainingDataSummary:
    """CSV의 계약과 이진 라벨 분포를 적은 메모리로 검증한다."""

    try:
        columns = pd.read_csv(data_path, nrows=0).columns
        if LABEL_COLUMN in columns:
            source = pd.read_csv(data_path, low_memory=False).reset_index(drop=True)
            dataset_kind = detect_training_dataset_kind(source.columns)
            target = validate_binary_target(source, context="training")
            if dataset_kind == "raw":
                preprocess_frame(source.loc[:, MODEL_INPUT_COLUMNS])
                datetimes = transaction_datetimes(source, context="training")
            else:
                validate_preprocessed_features(source)
                if transactions_path is None:
                    raise TrainingDatasetError(
                        "preprocessed training data requires companion transactions data"
                    )
                transaction_columns = pd.read_csv(
                    transactions_path, nrows=0
                ).columns
                missing_transactions = {
                    TRANSACTION_DATETIME_COLUMN,
                    LABEL_COLUMN,
                } - set(transaction_columns)
                if missing_transactions:
                    raise TrainingDatasetError(
                        "transactions data is missing required columns: "
                        f"{sorted(missing_transactions)}"
                    )
                transactions = pd.read_csv(
                    transactions_path,
                    usecols=[TRANSACTION_DATETIME_COLUMN, LABEL_COLUMN],
                    low_memory=False,
                )
                datetimes = aligned_transaction_datetimes(source, transactions)
            time_split_indices(
                datetimes,
                target,
                split_datetime=split_datetime,
                minimum_fraud_rows=minimum_fraud_rows_per_split,
            )

            fraud_count = int(target.sum())
            return TrainingDataSummary(
                row_count=len(target),
                column_count=len(columns),
                normal_count=len(target) - fraud_count,
                fraud_count=fraud_count,
                file_size_bytes=data_path.stat().st_size,
            )

        missing = {TARGET_COLUMN, GROUP_COLUMN} - set(columns)
        if missing:
            raise TrainingDataError(
                f"Required training columns are missing: {sorted(missing)}"
            )

        row_count = 0
        fraud_count = 0
        for chunk in pd.read_csv(
            data_path,
            usecols=[TARGET_COLUMN],
            chunksize=chunk_size,
            low_memory=False,
        ):
            target = binary_target(chunk[TARGET_COLUMN])
            row_count += len(target)
            fraud_count += int(target.sum())
    except TrainingDataError:
        raise
    except (TrainingDatasetError, FeaturePreprocessingError) as error:
        raise TrainingDataError(f"Invalid training data: {error}") from error
    except ValueError as error:
        raise TrainingDataError(f"Invalid training labels: {error}") from error
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise TrainingDataError(f"Failed to read training CSV: {data_path}") from error

    if row_count == 0:
        raise TrainingDataError(f"Training CSV contains no rows: {data_path}")

    return TrainingDataSummary(
        row_count=row_count,
        column_count=len(columns),
        normal_count=row_count - fraud_count,
        fraud_count=fraud_count,
        file_size_bytes=data_path.stat().st_size,
    )
