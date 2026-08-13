"""train1 raw64 데이터로 model80을 학습하는 Cloud Run Job 진입점."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from fdshield_ml.training.data_loader import (
    TrainingDataError,
    data_source_type,
    materialize_training_data,
)
from fdshield_ml.training.integrations.backend_callback import (
    TrainingCallbackConfig,
    TrainingResultNotificationError,
    notify_training_result,
    with_cloud_run_execution,
)
from fdshield_ml.training.service.train.train_service import (
    ProductionTrainingConfig,
    ProductionTrainingError,
    ProductionTrainingResult,
    ml_train_flow,
)

TrainingDataMaterializer = Callable[[str], AbstractContextManager[Path]]
ProductionTrainingRunner = Callable[
    [str | Path, str, ProductionTrainingConfig],
    ProductionTrainingResult,
]

DEFAULT_REGISTERED_MODEL_NAME = "fdshield-fraud-detector-v2"


@dataclass(frozen=True)
class TrainingJobConfig:
    """학습 Job 실행 전에 확정해야 하는 최소 설정."""

    data_uri: str
    experiment_name: str
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME
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
            "data_uri": environ.get("TRAINING_DATA_URI", "").strip(),
            "experiment_name": environ.get("MLFLOW_EXPERIMENT_NAME", "").strip(),
            "registered_model_name": environ.get(
                "MLFLOW_REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME
            ).strip(),
            "model_alias": environ.get("MLFLOW_MODEL_ALIAS", "champion").strip(),
            "cloud_run_execution_name": callback.cloud_run_execution_name,
        }

        missing = [name for name in ("data_uri", "experiment_name") if not values[name]]
        if missing:
            missing_variables = ", ".join(
                {
                    "data_uri": "TRAINING_DATA_URI",
                    "experiment_name": "MLFLOW_EXPERIMENT_NAME",
                }[name]
                for name in missing
            )
            raise ValueError(f"Required environment variables: {missing_variables}")

        if not values["registered_model_name"]:
            raise ValueError("MLFLOW_REGISTERED_MODEL_NAME must not be empty")
        if not values["model_alias"]:
            raise ValueError("MLFLOW_MODEL_ALIAS is required")

        data_source_type(values["data_uri"])
        try:
            minimum_pr_auc = float(environ.get("MODEL_MIN_PR_AUC", "0"))
            minimum_recall = float(environ.get("MODEL_MIN_RECALL", "0"))
        except ValueError as exc:
            raise ValueError(
                "MODEL_MIN_PR_AUC and MODEL_MIN_RECALL must be numbers"
            ) from exc

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
            with_cloud_run_execution(
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


def run_model_training(
    config: TrainingJobConfig,
    stream: TextIO = sys.stdout,
    materializer: TrainingDataMaterializer = materialize_training_data,
    training_runner: ProductionTrainingRunner = ml_train_flow,
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

    with materializer(config.data_uri) as data_path:
        result = training_runner(
            data_path,
            config.experiment_name,
            config.production_config(),
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
        with_cloud_run_execution(
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
        mode="train",
        status="success",
        model_version=result.model_version,
    )


def main(
    environ: Mapping[str, str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    materializer: TrainingDataMaterializer = materialize_training_data,
    training_runner: ProductionTrainingRunner = ml_train_flow,
    result_notifier: TrainingResultNotifier = notify_training_result,
) -> int:
    """환경변수를 검증하고 성공 여부를 프로세스 종료 코드로 반환한다."""

    config: TrainingJobConfig | None = None
    callback_config: TrainingCallbackConfig | None = None
    source_environ = os.environ if environ is None else environ
    try:
        callback_config = TrainingCallbackConfig.from_env(source_environ)
        config = TrainingJobConfig.from_env(source_environ)
        run_model_training(
            config,
            stdout,
            materializer,
            training_runner,
            result_notifier,
        )
    except TrainingResultNotificationError as error:
        _write_event(stderr, "training_result_notification_error", message=str(error))
        return 6
    except ProductionTrainingError as error:
        _notify_training_failure(
            config or callback_config, error, result_notifier, stderr
        )
        _write_event(stderr, "production_training_error", message=str(error))
        return 5
    except TrainingDataError as error:
        _notify_training_failure(
            config or callback_config, error, result_notifier, stderr
        )
        _write_event(stderr, "training_data_error", message=str(error))
        return 3
    except ValueError as error:
        _notify_training_failure(
            config or callback_config, error, result_notifier, stderr
        )
        _write_event(stderr, "training_job_configuration_error", message=str(error))
        return 2
    except Exception as error:  # noqa: BLE001 - process boundary reports failures
        _notify_training_failure(
            config or callback_config, error, result_notifier, stderr
        )
        _write_event(stderr, "unexpected_training_error", message=str(error))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
