"""Cloud Run Job에서 실행할 학습 작업의 초기 진입점.

Stub URI는 인프라 실행만 확인하고, 로컬 경로와 GCS URI는 학습 CSV를 준비해
필수 컬럼과 이진 라벨을 검증한다. 실제 학습 연결 전 데이터 입출력 경계를 먼저
확인하기 위한 단계다.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fdshield_ml.training.data_loader import (
    TrainingDataError,
    TrainingDataSummary,
    data_source_type,
    inspect_training_csv,
    materialize_training_data,
)
from fdshield_ml.training.dataset import DEFAULT_SPLIT_DATETIME
from fdshield_ml.training.job_tracking import (
    TrainingTrackingError,
    log_data_validation_run,
)
from fdshield_ml.training.production import (
    ProductionTrainingConfig,
    ProductionTrainingError,
    ProductionTrainingResult,
    train_and_register_model,
)


SUPPORTED_JOB_TYPES = frozenset({"binary"})
SUPPORTED_TRAINING_MODES = frozenset({"validate", "train"})
TrainingDataMaterializer = Callable[[str], AbstractContextManager[Path]]
ValidationRunLogger = Callable[
    [str, str, str, str, TrainingDataSummary],
    str,
]
ProductionTrainingRunner = Callable[
    [str | Path, str, ProductionTrainingConfig, str | Path | None],
    ProductionTrainingResult,
]

CALLBACK_MAX_ATTEMPTS = 3
CALLBACK_RETRY_DELAYS_SECONDS = (1.0, 2.0)


class TrainingResultNotificationError(RuntimeError):
    """Backend에 학습 결과를 기록하지 못했을 때 발생한다."""


@dataclass(frozen=True)
class TrainingCallbackConfig:
    """전체 학습 설정이 잘못돼도 Backend 실패 콜백에 필요한 최소 설정."""

    backend_training_run_id: int | None = None
    result_callback_url: str = ""
    result_callback_token: str = ""
    cloud_run_execution_name: str = ""

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> TrainingCallbackConfig:
        backend_training_run_id_value = environ.get(
            "BACKEND_TRAINING_RUN_ID", ""
        ).strip()
        callback_url = environ.get("TRAINING_RESULT_CALLBACK_URL", "").strip()
        callback_token = environ.get("TRAINING_RESULT_CALLBACK_TOKEN", "").strip()
        cloud_run_execution_name = environ.get("CLOUD_RUN_EXECUTION", "").strip()

        if not backend_training_run_id_value:
            return cls(cloud_run_execution_name=cloud_run_execution_name)

        try:
            backend_training_run_id = int(backend_training_run_id_value)
        except ValueError as exc:
            raise ValueError("BACKEND_TRAINING_RUN_ID must be an integer") from exc
        if backend_training_run_id < 1:
            raise ValueError("BACKEND_TRAINING_RUN_ID must be positive")
        if not callback_url or not callback_token:
            raise ValueError(
                "TRAINING_RESULT_CALLBACK_URL and TOKEN are required with "
                "BACKEND_TRAINING_RUN_ID"
            )

        return cls(
            backend_training_run_id=backend_training_run_id,
            result_callback_url=callback_url,
            result_callback_token=callback_token,
            cloud_run_execution_name=cloud_run_execution_name,
        )


@dataclass(frozen=True)
class TrainingJobConfig:
    """학습 Job 실행 전에 확정해야 하는 최소 설정."""

    job_type: str
    data_uri: str
    experiment_name: str
    transactions_uri: str = ""
    split_datetime: str = DEFAULT_SPLIT_DATETIME
    mode: str = "validate"
    registered_model_name: str = ""
    model_alias: str = "champion"
    minimum_pr_auc: float = 0.0
    minimum_recall: float = 0.0
    backend_training_run_id: int | None = None
    result_callback_url: str = ""
    result_callback_token: str = ""
    cloud_run_execution_name: str = ""

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> TrainingJobConfig:
        callback = TrainingCallbackConfig.from_env(environ)
        values = {
            "job_type": environ.get("TRAINING_JOB_TYPE", "").strip().lower(),
            "data_uri": environ.get("TRAINING_DATA_URI", "").strip(),
            "experiment_name": environ.get(
                "MLFLOW_EXPERIMENT_NAME", ""
            ).strip(),
            "transactions_uri": environ.get(
                "TRAINING_TRANSACTIONS_URI", ""
            ).strip(),
            "split_datetime": environ.get(
                "TRAINING_SPLIT_DATETIME", DEFAULT_SPLIT_DATETIME
            ).strip(),
            "mode": environ.get("TRAINING_MODE", "validate").strip().lower(),
            "registered_model_name": environ.get(
                "MLFLOW_REGISTERED_MODEL_NAME", ""
            ).strip(),
            "model_alias": environ.get("MLFLOW_MODEL_ALIAS", "champion").strip(),
            "cloud_run_execution_name": callback.cloud_run_execution_name,
        }

        missing = [
            name
            for name in ("job_type", "data_uri", "experiment_name")
            if not values[name]
        ]
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
        if values["mode"] not in SUPPORTED_TRAINING_MODES:
            supported = ", ".join(sorted(SUPPORTED_TRAINING_MODES))
            raise ValueError(f"TRAINING_MODE must be one of: {supported}")
        if values["mode"] == "train" and not values["registered_model_name"]:
            raise ValueError("MLFLOW_REGISTERED_MODEL_NAME is required in train mode")
        if not values["model_alias"]:
            raise ValueError("MLFLOW_MODEL_ALIAS is required")

        data_source_type(values["data_uri"])
        if values["mode"] == "train" and data_source_type(values["data_uri"]) == "stub":
            raise ValueError("TRAINING_MODE=train requires a local or gs:// data URI")
        if values["transactions_uri"]:
            if data_source_type(values["transactions_uri"]) == "stub":
                raise ValueError(
                    "TRAINING_TRANSACTIONS_URI requires a local or gs:// data URI"
                )
        try:
            datetime.fromisoformat(values["split_datetime"])
        except ValueError as exc:
            raise ValueError(
                "TRAINING_SPLIT_DATETIME must be an ISO datetime"
            ) from exc

        try:
            minimum_pr_auc = float(environ.get("MODEL_MIN_PR_AUC", "0"))
            minimum_recall = float(environ.get("MODEL_MIN_RECALL", "0"))
        except ValueError as exc:
            raise ValueError("MODEL_MIN_PR_AUC and MODEL_MIN_RECALL must be numbers") from exc

        return cls(
            **values,
            minimum_pr_auc=minimum_pr_auc,
            minimum_recall=minimum_recall,
            backend_training_run_id=callback.backend_training_run_id,
            result_callback_url=callback.result_callback_url,
            result_callback_token=callback.result_callback_token,
        )

    def production_config(self) -> ProductionTrainingConfig:
        return ProductionTrainingConfig(
            registered_model_name=self.registered_model_name,
            model_alias=self.model_alias,
            minimum_pr_auc=self.minimum_pr_auc,
            minimum_recall=self.minimum_recall,
            split_datetime=self.split_datetime,
        )


CallbackConfig = TrainingJobConfig | TrainingCallbackConfig
TrainingResultNotifier = Callable[[CallbackConfig, dict[str, object]], None]


def _write_event(stream: TextIO, event: str, **fields: object) -> None:
    """Cloud Logging에서 검색하기 쉬운 한 줄 JSON 로그를 출력한다."""

    print(
        json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True),
        file=stream,
        flush=True,
    )


def _config_log_fields(config: TrainingJobConfig) -> dict[str, object]:
    """구조화 로그에서 Backend 관리 토큰을 제외한다."""

    fields = asdict(config)
    fields.pop("result_callback_token", None)
    return fields


def notify_training_result(
    config: CallbackConfig,
    payload: dict[str, object],
    *,
    opener: Callable[..., object] = urlopen,
    sleeper: Callable[[float], None] = sleep,
) -> None:
    """Backend에 최소 학습 결과를 기록하고 일시적 실패는 재시도한다."""

    if config.backend_training_run_id is None:
        return
    url = config.result_callback_url.format(
        training_run_id=config.backend_training_run_id
    )
    encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):
        request = Request(
            url,
            data=encoded_payload,
            headers={
                "Content-Type": "application/json",
                "X-MLOps-Admin-Token": config.result_callback_token,
            },
            method="POST",
        )
        try:
            with opener(request, timeout=20) as response:  # type: ignore[attr-defined]
                status = response.status  # type: ignore[attr-defined]
                if 200 <= status < 300:
                    return
                if not _is_retryable_callback_status(status):
                    raise TrainingResultNotificationError(
                        f"Backend callback returned HTTP {status}."
                    )
        except HTTPError as exc:
            if not _is_retryable_callback_status(exc.code):
                raise TrainingResultNotificationError(
                    f"Backend callback returned HTTP {exc.code}."
                ) from exc
            last_error: BaseException = exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
        else:
            last_error = TrainingResultNotificationError(
                "Backend callback returned a retryable HTTP status."
            )

        if attempt < CALLBACK_MAX_ATTEMPTS:
            sleeper(CALLBACK_RETRY_DELAYS_SECONDS[attempt - 1])

    raise TrainingResultNotificationError(
        "Failed to record the training result in Backend after retries."
    ) from last_error


def _is_retryable_callback_status(status: int) -> bool:
    """재호출해도 안전한 일시적 HTTP 오류인지 판별한다."""

    return status in {408, 429} or status >= 500


def _with_cloud_run_execution(
    config: CallbackConfig,
    payload: dict[str, object],
) -> dict[str, object]:
    """Cloud Run이 자동 제공한 실행 이름을 설정된 경우에만 추가한다."""

    if not config.cloud_run_execution_name:
        return payload
    return {
        **payload,
        "cloud_run_execution_name": config.cloud_run_execution_name,
    }


def _notify_training_failure(
    config: CallbackConfig | None,
    error: BaseException,
    result_notifier: TrainingResultNotifier,
    stderr: TextIO,
) -> None:
    """가능한 경우 원래 실패 코드를 보존하며 Backend 상태를 FAILED로 종결한다."""

    if config is None or config.backend_training_run_id is None:
        return
    try:
        result_notifier(
            config,
            _with_cloud_run_execution(
                config,
                {
                    "status": "FAILED",
                    "error_message": str(error)[:1000],
                },
            ),
        )
    except TrainingResultNotificationError as callback_error:
        _write_event(
            stderr,
            "training_result_notification_error",
            message=str(callback_error),
        )


def run_stub(config: TrainingJobConfig, stream: TextIO = sys.stdout) -> None:
    """인프라 실행 흐름만 확인하고 실제 모델 학습은 수행하지 않는다."""

    _write_event(stream, "training_job_started", **_config_log_fields(config))
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
        **_config_log_fields(config),
        source_type=source_type,
    )

    with ExitStack() as stack:
        data_path = stack.enter_context(materializer(config.data_uri))
        transactions_path = (
            stack.enter_context(materializer(config.transactions_uri))
            if config.transactions_uri
            else None
        )
        summary = inspect_training_csv(
            data_path,
            transactions_path=transactions_path,
            split_datetime=config.split_datetime,
        )

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


def run_model_training(
    config: TrainingJobConfig,
    stream: TextIO = sys.stdout,
    materializer: TrainingDataMaterializer = materialize_training_data,
    training_runner: ProductionTrainingRunner = train_and_register_model,
    result_notifier: TrainingResultNotifier = notify_training_result,
) -> None:
    """학습 데이터를 내려받아 실제 모델을 학습·등록하고 결과를 기록한다."""

    source_type = data_source_type(config.data_uri)
    _write_event(
        stream,
        "training_job_started",
        **_config_log_fields(config),
        source_type=source_type,
    )

    with ExitStack() as stack:
        data_path = stack.enter_context(materializer(config.data_uri))
        transactions_path = (
            stack.enter_context(materializer(config.transactions_uri))
            if config.transactions_uri
            else None
        )
        result = training_runner(
            data_path,
            config.experiment_name,
            config.production_config(),
            transactions_path,
        )
    _write_event(
        stream,
        "model_registered",
        run_id=result.run_id,
        registered_model_name=config.registered_model_name,
        model_version=result.model_version,
        validation_passed=result.validation_passed,
        metrics=result.metrics,
        recommendation=result.recommendation,
        champion_model_version=result.champion_model_version,
        champion_metrics=result.champion_metrics,
    )
    result_notifier(
        config,
        _with_cloud_run_execution(
            config,
            {
                "status": "SUCCEEDED",
                "mlflow_run_id": result.run_id,
            },
        ),
    )
    _write_event(
        stream,
        "training_job_completed",
        job_type=config.job_type,
        mode="train",
        status="success",
        model_version=result.model_version,
    )


def main(
    environ: Mapping[str, str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    materializer: TrainingDataMaterializer = materialize_training_data,
    validation_run_logger: ValidationRunLogger = log_data_validation_run,
    training_runner: ProductionTrainingRunner = train_and_register_model,
    result_notifier: TrainingResultNotifier = notify_training_result,
) -> int:
    """환경변수를 검증하고 성공 여부를 프로세스 종료 코드로 반환한다."""

    config: TrainingJobConfig | None = None
    callback_config: TrainingCallbackConfig | None = None
    source_environ = os.environ if environ is None else environ
    try:
        callback_config = TrainingCallbackConfig.from_env(source_environ)
        config = TrainingJobConfig.from_env(source_environ)
        if config.mode == "train":
            run_model_training(
                config,
                stdout,
                materializer,
                training_runner,
                result_notifier,
            )
        elif data_source_type(config.data_uri) == "stub":
            run_stub(config, stdout)
        else:
            run_data_validation(
                config,
                stdout,
                materializer,
                validation_run_logger,
            )
    except TrainingResultNotificationError as error:
        _write_event(stderr, "training_result_notification_error", message=str(error))
        return 6
    except TrainingTrackingError as error:
        _notify_training_failure(config or callback_config, error, result_notifier, stderr)
        _write_event(stderr, "training_tracking_error", message=str(error))
        return 4
    except ProductionTrainingError as error:
        _notify_training_failure(config or callback_config, error, result_notifier, stderr)
        _write_event(stderr, "production_training_error", message=str(error))
        return 5
    except TrainingDataError as error:
        _notify_training_failure(config or callback_config, error, result_notifier, stderr)
        _write_event(stderr, "training_data_error", message=str(error))
        return 3
    except ValueError as error:
        _notify_training_failure(config or callback_config, error, result_notifier, stderr)
        _write_event(stderr, "training_job_configuration_error", message=str(error))
        return 2
    except Exception as error:
        _notify_training_failure(config or callback_config, error, result_notifier, stderr)
        _write_event(stderr, "unexpected_training_error", message=str(error))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
