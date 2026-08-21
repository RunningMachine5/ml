"""실제 모델 학습만 수행하는 Cloud Run Training Job 진입점 테스트."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Self
from urllib.error import HTTPError, URLError

import pytest

from fdshield_ml.infrastructure.training_pipeline import (
    ProductionTrainingConfig,
    ProductionTrainingError,
    ProductionTrainingResult,
)
from fdshield_ml.training_job import (
    DEFAULT_REGISTERED_MODEL_NAME,
    TrainingJobConfig,
    TrainingResultNotificationError,
    main,
    notify_training_result,
)

VALID_ENV = {
    "TRAINING_DATA_URI": "data/open/train1.csv",
    "MLFLOW_EXPERIMENT_NAME": "fdshield-binary-training",
}


def _successful_result() -> ProductionTrainingResult:
    return ProductionTrainingResult(
        run_id="run-123",
        model_version=17,
        metrics={"validation_pr_auc": 0.9, "validation_recall": 0.85},
        validation_passed=True,
        recommendation="RECOMMENDED",
        champion_model_version=1,
        champion_metrics={
            "validation_pr_auc": 0.8,
            "validation_recall": 0.8,
            "validation_fpr": 0.01,
        },
    )


def test_training_job_config_reads_required_environment() -> None:
    config = TrainingJobConfig.from_env(VALID_ENV)

    assert config == TrainingJobConfig(
        data_uri="data/open/train1.csv",
        experiment_name="fdshield-binary-training",
        registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
    )


@pytest.mark.parametrize(
    "variable",
    ["TRAINING_DATA_URI", "MLFLOW_EXPERIMENT_NAME"],
)
def test_training_job_config_rejects_missing_environment(variable: str) -> None:
    environ = {**VALID_ENV, variable: ""}

    with pytest.raises(ValueError, match=variable):
        TrainingJobConfig.from_env(environ)


def test_training_job_config_rejects_unsupported_data_uri() -> None:
    environ = {**VALID_ENV, "TRAINING_DATA_URI": "https://example.com/train.csv"}

    with pytest.raises(ValueError, match="TRAINING_DATA_URI"):
        TrainingJobConfig.from_env(environ)


def test_training_job_config_rejects_empty_registered_model() -> None:
    environ = {**VALID_ENV, "MLFLOW_REGISTERED_MODEL_NAME": ""}

    with pytest.raises(ValueError, match="MLFLOW_REGISTERED_MODEL_NAME"):
        TrainingJobConfig.from_env(environ)


def test_training_job_config_builds_candidate_comparison_policy() -> None:
    environ = {
        **VALID_ENV,
        "MLFLOW_REGISTERED_MODEL_NAME": "fdshield-fraud-detector-v2",
        "MLFLOW_MODEL_ALIAS": "champion",
        "MODEL_MIN_PR_AUC": "0.75",
        "MODEL_MIN_RECALL": "0.8",
    }

    config = TrainingJobConfig.from_env(environ)

    assert config.production_config() == ProductionTrainingConfig(
        registered_model_name="fdshield-fraud-detector-v2",
        model_alias="champion",
        minimum_pr_auc=0.75,
        minimum_recall=0.8,
    )


def test_training_job_main_reports_configuration_error() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main({}, stdout=stdout, stderr=stderr)
    error = json.loads(stderr.getvalue())

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert error["event"] == "training_job_configuration_error"
    assert "TRAINING_DATA_URI" in error["message"]


def test_training_job_configuration_failure_notifies_backend() -> None:
    environ = {
        **VALID_ENV,
        "TRAINING_DATA_URI": "",
        "BACKEND_TRAINING_RUN_ID": "7",
        "CLOUD_RUN_EXECUTION": "fdshield-binary-training-config1",
        "TRAINING_RESULT_CALLBACK_URL": (
            "https://api.example/mlops/training/runs/{training_run_id}/result"
        ),
        "TRAINING_RESULT_CALLBACK_TOKEN": "callback-secret",
    }
    notifications: list[dict[str, object]] = []

    exit_code = main(
        environ,
        stderr=io.StringIO(),
        result_notifier=lambda config, payload: notifications.append(payload),
    )

    assert exit_code == 2
    assert notifications == [
        {
            "status": "FAILED",
            "error_message": "Required environment variables: TRAINING_DATA_URI",
            "cloud_run_execution_name": "fdshield-binary-training-config1",
        }
    ]


def test_training_job_main_reports_candidate_result(tmp_path: Path) -> None:
    source = tmp_path / "train1.csv"
    source.write_text("placeholder", encoding="utf-8")
    environ = {
        **VALID_ENV,
        "TRAINING_DATA_URI": str(source),
        "BACKEND_TRAINING_RUN_ID": "7",
        "CLOUD_RUN_EXECUTION": "fdshield-binary-training-abc12",
        "TRAINING_RESULT_CALLBACK_URL": (
            "https://api.example/mlops/training/runs/{training_run_id}/result"
        ),
        "TRAINING_RESULT_CALLBACK_TOKEN": "do-not-log-this-token",
    }
    stdout = io.StringIO()
    stderr = io.StringIO()
    notifications: list[dict[str, object]] = []

    def successful_training_runner(
        data_path: str | Path,
        experiment_name: str,
        config: ProductionTrainingConfig,
    ) -> ProductionTrainingResult:
        assert Path(data_path) == source
        assert experiment_name == "fdshield-binary-training"
        assert config.registered_model_name == "fdshield-fraud-detector-v2"
        return _successful_result()

    exit_code = main(
        environ,
        stdout=stdout,
        stderr=stderr,
        training_runner=successful_training_runner,
        result_notifier=lambda config, payload: notifications.append(payload),
    )
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "do-not-log-this-token" not in stdout.getvalue()
    assert [event["event"] for event in events] == [
        "training_job_started",
        "model_registered",
        "training_job_completed",
    ]
    assert events[1]["model_version"] == 17
    assert events[1]["recommendation"] == "RECOMMENDED"
    assert notifications == [
        {
            "status": "RUNNING",
            "cloud_run_execution_name": "fdshield-binary-training-abc12",
        },
        {
            "status": "SUCCEEDED",
            "mlflow_run_id": "run-123",
            "cloud_run_execution_name": "fdshield-binary-training-abc12",
        }
    ]


def test_training_result_callback_retries_same_payload_on_transient_error() -> None:
    config = TrainingJobConfig.from_env(
        {
            **VALID_ENV,
            "BACKEND_TRAINING_RUN_ID": "7",
            "TRAINING_RESULT_CALLBACK_URL": (
                "https://api.example/mlops/training/runs/{training_run_id}/result"
            ),
            "TRAINING_RESULT_CALLBACK_TOKEN": "callback-secret",
        }
    )
    attempts: list[dict[str, object]] = []
    delays: list[float] = []

    class SuccessfulResponse:
        status = 204

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def flaky_opener(request: object, timeout: int) -> SuccessfulResponse:
        assert timeout == 20
        attempts.append(json.loads(request.data))  # type: ignore[attr-defined]
        if len(attempts) < 3:
            raise URLError("temporary unavailable")
        return SuccessfulResponse()

    payload: dict[str, object] = {
        "status": "SUCCEEDED",
        "mlflow_run_id": "run-123",
    }
    notify_training_result(
        config,
        payload,
        opener=flaky_opener,
        sleeper=delays.append,
    )

    assert attempts == [payload, payload, payload]
    assert delays == [1.0, 2.0]


def test_training_result_callback_does_not_retry_conflict() -> None:
    config = TrainingJobConfig.from_env(
        {
            **VALID_ENV,
            "BACKEND_TRAINING_RUN_ID": "7",
            "TRAINING_RESULT_CALLBACK_URL": (
                "https://api.example/mlops/training/runs/{training_run_id}/result"
            ),
            "TRAINING_RESULT_CALLBACK_TOKEN": "callback-secret",
        }
    )
    attempts = 0

    def conflict_opener(request: object, timeout: int) -> object:
        nonlocal attempts
        attempts += 1
        raise HTTPError(
            "https://api.example/mlops/training/runs/7/result",
            409,
            "Conflict",
            hdrs=None,
            fp=None,
        )

    with pytest.raises(TrainingResultNotificationError, match="HTTP 409"):
        notify_training_result(
            config,
            {"status": "FAILED"},
            opener=conflict_opener,
            sleeper=lambda _: None,
        )

    assert attempts == 1


def test_training_data_failure_notifies_backend(tmp_path: Path) -> None:
    environ = {
        **VALID_ENV,
        "TRAINING_DATA_URI": str(tmp_path / "missing.csv"),
        "BACKEND_TRAINING_RUN_ID": "7",
        "TRAINING_RESULT_CALLBACK_URL": (
            "https://api.example/mlops/training/runs/{training_run_id}/result"
        ),
        "TRAINING_RESULT_CALLBACK_TOKEN": "callback-secret",
    }
    notifications: list[dict[str, object]] = []

    exit_code = main(
        environ,
        stderr=io.StringIO(),
        result_notifier=lambda config, payload: notifications.append(payload),
    )

    assert exit_code == 3
    assert notifications[0]["status"] == "FAILED"
    assert "missing.csv" in str(notifications[0]["error_message"])


def test_production_training_failure_notifies_backend(tmp_path: Path) -> None:
    source = tmp_path / "train1.csv"
    source.write_text("placeholder", encoding="utf-8")
    environ = {
        **VALID_ENV,
        "TRAINING_DATA_URI": str(source),
        "BACKEND_TRAINING_RUN_ID": "7",
        "CLOUD_RUN_EXECUTION": "fdshield-binary-training-failed1",
        "TRAINING_RESULT_CALLBACK_URL": (
            "https://api.example/mlops/training/runs/{training_run_id}/result"
        ),
        "TRAINING_RESULT_CALLBACK_TOKEN": "callback-secret",
    }
    notifications: list[dict[str, object]] = []

    def failed_training_runner(*args: object) -> ProductionTrainingResult:
        raise ProductionTrainingError("MLflow registration failed")

    exit_code = main(
        environ,
        stderr=io.StringIO(),
        training_runner=failed_training_runner,
        result_notifier=lambda config, payload: notifications.append(payload),
    )

    assert exit_code == 5
    assert notifications == [
        {
            "status": "RUNNING",
            "cloud_run_execution_name": "fdshield-binary-training-failed1",
        },
        {
            "status": "FAILED",
            "error_message": "MLflow registration failed",
            "cloud_run_execution_name": "fdshield-binary-training-failed1",
        }
    ]


def test_unexpected_training_failure_notifies_backend(tmp_path: Path) -> None:
    source = tmp_path / "train1.csv"
    source.write_text("placeholder", encoding="utf-8")
    environ = {
        **VALID_ENV,
        "TRAINING_DATA_URI": str(source),
        "BACKEND_TRAINING_RUN_ID": "7",
        "TRAINING_RESULT_CALLBACK_URL": (
            "https://api.example/mlops/training/runs/{training_run_id}/result"
        ),
        "TRAINING_RESULT_CALLBACK_TOKEN": "callback-secret",
    }
    notifications: list[dict[str, object]] = []
    stderr = io.StringIO()

    def crashed_training_runner(*args: object) -> ProductionTrainingResult:
        raise RuntimeError("unexpected xgboost failure")

    exit_code = main(
        environ,
        stderr=stderr,
        training_runner=crashed_training_runner,
        result_notifier=lambda config, payload: notifications.append(payload),
    )

    assert exit_code == 1
    assert notifications[0]["status"] == "FAILED"
    assert json.loads(stderr.getvalue())["event"] == "unexpected_training_error"
