"""로컬 파일과 GCS 객체를 동일한 train1 학습 데이터 경로로 다룬다."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from google.cloud import storage

DataSourceType = Literal["gcs", "local"]
StorageClientFactory = Callable[[], Any]


class TrainingDataError(RuntimeError):
    """학습 데이터 경로, 다운로드 또는 train1 CSV 검증 실패."""


@dataclass(frozen=True)
class GCSObjectLocation:
    """검증된 GCS Bucket과 객체 이름."""

    bucket_name: str
    object_name: str

    @classmethod
    def from_uri(cls, uri: str) -> GCSObjectLocation:
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


def data_source_type(data_uri: str) -> DataSourceType:
    """환경변수 값이 GCS URI 또는 로컬 경로인지 구분한다."""

    normalized = data_uri.strip()
    lowered = normalized.lower()

    if lowered.startswith("gs://"):
        GCSObjectLocation.from_uri(normalized)
        return "gcs"
    if "://" in normalized:
        raise ValueError("TRAINING_DATA_URI must be a local path or gs:// URI.")
    return "local"


@contextmanager
def materialize_training_data(
    data_uri: str,
    *,
    storage_client_factory: StorageClientFactory = storage.Client,
) -> Iterator[Path]:
    """로컬 경로는 그대로 사용하고 GCS 객체만 임시 파일로 다운로드한다."""

    source_type = data_source_type(data_uri)
    if source_type == "local":
        local_path = Path(data_uri).expanduser().resolve()
        if not local_path.is_file():
            raise TrainingDataError(f"Training CSV not found: {local_path}")
        yield local_path
        return

    location = GCSObjectLocation.from_uri(data_uri)
    file_name = Path(location.object_name).name or "train1.csv"

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


__all__ = [
    "GCSObjectLocation",
    "TrainingDataError",
    "data_source_type",
    "materialize_training_data",
]
