"""Cloud Run Job에서 실행할 학습 작업의 초기 진입점.

현재는 인프라 연결을 검증하기 위한 Stub만 실행한다. 실제 학습을 연결할 때는
검증된 설정을 기존 ``train_xgboost`` 학습 흐름에 전달하도록 교체한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import sys
from collections.abc import Mapping
from typing import TextIO
from urllib.parse import urlparse


SUPPORTED_JOB_TYPES = frozenset({"binary"})
SUPPORTED_DATA_SCHEMES = frozenset({"gs", "stub"})


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

        data_scheme = urlparse(values["data_uri"]).scheme.lower()
        if data_scheme not in SUPPORTED_DATA_SCHEMES:
            supported = ", ".join(sorted(SUPPORTED_DATA_SCHEMES))
            raise ValueError(f"TRAINING_DATA_URI scheme must be one of: {supported}")

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


def main(
    environ: Mapping[str, str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """환경변수를 검증하고 성공 여부를 프로세스 종료 코드로 반환한다."""

    try:
        config = TrainingJobConfig.from_env(os.environ if environ is None else environ)
        run_stub(config, stdout)
    except ValueError as error:
        _write_event(stderr, "training_job_configuration_error", message=str(error))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
