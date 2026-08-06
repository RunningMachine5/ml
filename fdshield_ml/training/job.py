"""Cloud Run Job에서 실행할 학습 작업의 초기 진입점.

Stub URI는 인프라 실행만 확인하고, 로컬 경로와 GCS URI는 학습 CSV를 준비해
필수 컬럼과 이진 라벨을 검증한다. 실제 학습 연결 전 데이터 입출력 경계를 먼저
확인하기 위한 단계다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import TextIO

from fdshield_ml.training.data_loader import (
    TrainingDataError,
    TrainingDataSummary,
    data_source_type,
    inspect_training_csv,
    materialize_training_data,
)
from fdshield_ml.training.job_tracking import (
    TrainingTrackingError,
    log_data_validation_run,
)


SUPPORTED_JOB_TYPES = frozenset({"binary"})
TrainingDataMaterializer = Callable[[str], AbstractContextManager[Path]]
ValidationRunLogger = Callable[
    [str, str, str, str, TrainingDataSummary],
    str,
]


@dataclass(frozen=True)
class TrainingJobConfig:
    """학습 Job 실행 전에 확정해야 하는 최소 설정."""

    job_type: str
    data_uri: str
    experiment_name: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "TrainingJobConfig":
        values = {
            "job_type": environ.get("TRAINING_JOB_TYPE", "").strip().lower(),
            "data_uri": environ.get("TRAINING_DATA_URI", "").strip(),
            "experiment_name": environ.get(
                "MLFLOW_EXPERIMENT_NAME", ""
            ).strip(),
        }

        missing = [name for name, value in values.items() if not value]
        if missing:
            missing_variables = ", ".join(
                {
                    "job_type": "TRAINING_JOB_TYPE",
                    "data_uri": "TRAINING_DATA_URI",
                    "experiment_name": "MLFLOW_EXPERIMENT_NAME",
                }[name]
                for name in missing
            )
            raise ValueError(f"Required environment variables: {missing_variables}")

        if values["job_type"] not in SUPPORTED_JOB_TYPES:
            supported = ", ".join(sorted(SUPPORTED_JOB_TYPES))
            raise ValueError(f"TRAINING_JOB_TYPE must be one of: {supported}")

        data_source_type(values["data_uri"])

        return cls(**values)


def _write_event(stream: TextIO, event: str, **fields: object) -> None:
    """Cloud Logging에서 검색하기 쉬운 한 줄 JSON 로그를 출력한다."""

    print(
        json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True),
        file=stream,
        flush=True,
    )


def run_stub(config: TrainingJobConfig, stream: TextIO = sys.stdout) -> None:
    """인프라 실행 흐름만 확인하고 실제 모델 학습은 수행하지 않는다."""

    _write_event(stream, "training_job_started", **asdict(config))
    _write_event(
        stream,
        "training_job_completed",
        job_type=config.job_type,
        mode="stub",
        status="success",
    )


def run_data_validation(
    config: TrainingJobConfig,
    stream: TextIO = sys.stdout,
    materializer: TrainingDataMaterializer = materialize_training_data,
    validation_run_logger: ValidationRunLogger = log_data_validation_run,
) -> None:
    """로컬 또는 GCS CSV를 검증하고 요약을 MLflow에 기록한다."""

    source_type = data_source_type(config.data_uri)
    _write_event(
        stream,
        "training_job_started",
        **asdict(config),
        source_type=source_type,
    )

    with materializer(config.data_uri) as data_path:
        summary = inspect_training_csv(data_path)

    _write_event(
        stream,
        "training_data_validated",
        **asdict(summary),
        data_uri=config.data_uri,
        source_type=source_type,
    )
    run_id = validation_run_logger(
        config.job_type,
        config.data_uri,
        config.experiment_name,
        source_type,
        summary,
    )
    _write_event(
        stream,
        "training_validation_tracked",
        experiment_name=config.experiment_name,
        run_id=run_id,
    )
    _write_event(
        stream,
        "training_job_completed",
        job_type=config.job_type,
        mode="data-validation",
        status="success",
    )


def main(
    environ: Mapping[str, str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    materializer: TrainingDataMaterializer = materialize_training_data,
    validation_run_logger: ValidationRunLogger = log_data_validation_run,
) -> int:
    """환경변수를 검증하고 성공 여부를 프로세스 종료 코드로 반환한다."""

    try:
        config = TrainingJobConfig.from_env(os.environ if environ is None else environ)
        if data_source_type(config.data_uri) == "stub":
            run_stub(config, stdout)
        else:
            run_data_validation(
                config,
                stdout,
                materializer,
                validation_run_logger,
            )
    except TrainingTrackingError as error:
        _write_event(stderr, "training_tracking_error", message=str(error))
        return 4
    except TrainingDataError as error:
        _write_event(stderr, "training_data_error", message=str(error))
        return 3
    except ValueError as error:
        _write_event(stderr, "training_job_configuration_error", message=str(error))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
