"""Cloud Run Training Job 진입점 테스트."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from fdshield_ml.training.data_loader import TrainingDataSummary
from fdshield_ml.training.job import TrainingJobConfig, main
from fdshield_ml.training.job_tracking import TrainingTrackingError
from fdshield_ml.training.production import (
    ProductionTrainingConfig,
    ProductionTrainingResult,
)


VALID_ENV = {
    "TRAINING_JOB_TYPE": "binary",
    "TRAINING_DATA_URI": "stub://local-data",
    "MLFLOW_EXPERIMENT_NAME": "fdshield-binary-training",
}


def successful_validation_run_logger(
    job_type: str,
    data_uri: str,
    experiment_name: str,
    source_type: str,
    summary: TrainingDataSummary,
) -> str:
    return "test-run-id"


def test_training_job_config_reads_required_environment() -> None:
    config = TrainingJobConfig.from_env(VALID_ENV)

    assert config == TrainingJobConfig(
        job_type="binary",
        data_uri="stub://local-data",
        experiment_name="fdshield-binary-training",
    )


@pytest.mark.parametrize(
    "variable",
    ["TRAINING_JOB_TYPE", "TRAINING_DATA_URI", "MLFLOW_EXPERIMENT_NAME"],
)
def test_training_job_config_rejects_missing_environment(variable: str) -> None:
    environ = {**VALID_ENV, variable: ""}

    with pytest.raises(ValueError, match=variable):
        TrainingJobConfig.from_env(environ)


def test_training_job_config_rejects_unsupported_job_type() -> None:
    environ = {**VALID_ENV, "TRAINING_JOB_TYPE": "fraud-type"}

    with pytest.raises(ValueError, match="TRAINING_JOB_TYPE"):
        TrainingJobConfig.from_env(environ)


def test_training_job_config_rejects_unsupported_data_uri() -> None:
    environ = {**VALID_ENV, "TRAINING_DATA_URI": "https://example.com/train.csv"}

    with pytest.raises(ValueError, match="TRAINING_DATA_URI"):
        TrainingJobConfig.from_env(environ)


def test_training_job_config_accepts_local_data_path() -> None:
    environ = {**VALID_ENV, "TRAINING_DATA_URI": "data/open/train.csv"}

    assert TrainingJobConfig.from_env(environ).data_uri == "data/open/train.csv"


def test_training_job_config_reads_companion_uri_and_split_datetime() -> None:
    environ = {
        **VALID_ENV,
        "TRAINING_TRANSACTIONS_URI": "data/generated/transactions.csv",
        "TRAINING_SPLIT_DATETIME": "2026-05-01 00:00:00",
    }

    config = TrainingJobConfig.from_env(environ)

    assert config.transactions_uri == "data/generated/transactions.csv"
    assert config.split_datetime == "2026-05-01 00:00:00"


def test_training_job_config_rejects_invalid_split_datetime() -> None:
    environ = {**VALID_ENV, "TRAINING_SPLIT_DATETIME": "not-a-datetime"}

    with pytest.raises(ValueError, match="TRAINING_SPLIT_DATETIME"):
        TrainingJobConfig.from_env(environ)


def test_training_job_config_requires_registered_model_in_train_mode() -> None:
    environ = {
        **VALID_ENV,
        "TRAINING_MODE": "train",
        "TRAINING_DATA_URI": "data/train.csv",
    }

    with pytest.raises(ValueError, match="MLFLOW_REGISTERED_MODEL_NAME"):
        TrainingJobConfig.from_env(environ)


def test_training_job_config_builds_production_gate() -> None:
    environ = {
        **VALID_ENV,
        "TRAINING_MODE": "train",
        "TRAINING_DATA_URI": "data/train.csv",
        "MLFLOW_REGISTERED_MODEL_NAME": "fdshield-fraud-detector",
        "MLFLOW_MODEL_ALIAS": "champion",
        "MLFLOW_AUTO_PROMOTE": "true",
        "MODEL_MIN_PR_AUC": "0.75",
        "MODEL_MIN_RECALL": "0.8",
    }

    config = TrainingJobConfig.from_env(environ)

    assert config.production_config() == ProductionTrainingConfig(
        registered_model_name="fdshield-fraud-detector",
        model_alias="champion",
        auto_promote=True,
        minimum_pr_auc=0.75,
        minimum_recall=0.8,
    )


def test_training_job_main_emits_structured_events_and_succeeds() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(VALID_ENV, stdout=stdout, stderr=stderr)
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert [event["event"] for event in events] == [
        "training_job_started",
        "training_job_completed",
    ]
    assert events[0]["data_uri"] == "stub://local-data"
    assert events[1]["status"] == "success"


def test_training_job_main_reports_configuration_error() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main({}, stdout=stdout, stderr=stderr)
    error = json.loads(stderr.getvalue())

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert error["event"] == "training_job_configuration_error"


def test_training_job_main_validates_local_csv(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Account_account_number": ["account-1", "account-2"],
            "Transaction_Amount": [10_000, 50_000],
            "Fraud_Type": ["m", "a"],
        }
    ).to_csv(source, index=False)
    environ = {**VALID_ENV, "TRAINING_DATA_URI": str(source)}
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        environ,
        stdout=stdout,
        stderr=stderr,
        validation_run_logger=successful_validation_run_logger,
    )
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert [event["event"] for event in events] == [
        "training_job_started",
        "training_data_validated",
        "training_validation_tracked",
        "training_job_completed",
    ]
    assert events[0]["source_type"] == "local"
    assert events[1]["row_count"] == 2
    assert events[1]["normal_count"] == 1
    assert events[1]["fraud_count"] == 1
    assert events[2]["experiment_name"] == "fdshield-binary-training"
    assert events[2]["run_id"] == "test-run-id"
    assert events[3]["mode"] == "data-validation"


def test_training_job_main_reports_data_error() -> None:
    environ = {**VALID_ENV, "TRAINING_DATA_URI": "missing/train.csv"}
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(environ, stdout=stdout, stderr=stderr)
    error = json.loads(stderr.getvalue())
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert exit_code == 3
    assert [event["event"] for event in events] == ["training_job_started"]
    assert events[0]["source_type"] == "local"
    assert error["event"] == "training_data_error"


def test_training_job_main_reports_tracking_error(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Account_account_number": ["account-1", "account-2"],
            "Fraud_Type": ["m", "a"],
        }
    ).to_csv(source, index=False)
    environ = {**VALID_ENV, "TRAINING_DATA_URI": str(source)}
    stdout = io.StringIO()
    stderr = io.StringIO()

    def failed_validation_run_logger(
        job_type: str,
        data_uri: str,
        experiment_name: str,
        source_type: str,
        summary: TrainingDataSummary,
    ) -> str:
        raise TrainingTrackingError("MLflow is unavailable.")

    exit_code = main(
        environ,
        stdout=stdout,
        stderr=stderr,
        validation_run_logger=failed_validation_run_logger,
    )
    error = json.loads(stderr.getvalue())

    assert exit_code == 4
    assert error["event"] == "training_tracking_error"
    assert "MLflow is unavailable" in error["message"]


def test_training_job_main_trains_registers_and_promotes_model(tmp_path: Path) -> None:
    source = tmp_path / "transactions.csv"
    source.write_text("placeholder", encoding="utf-8")
    companion = tmp_path / "raw-transactions.csv"
    companion.write_text("placeholder", encoding="utf-8")
    environ = {
        **VALID_ENV,
        "TRAINING_MODE": "train",
        "TRAINING_DATA_URI": str(source),
        "TRAINING_TRANSACTIONS_URI": str(companion),
        "TRAINING_SPLIT_DATETIME": "2026-05-01 00:00:00",
        "MLFLOW_REGISTERED_MODEL_NAME": "fdshield-fraud-detector",
        "MLFLOW_AUTO_PROMOTE": "true",
    }
    stdout = io.StringIO()
    stderr = io.StringIO()

    def successful_training_runner(
        data_path: str | Path,
        experiment_name: str,
        config: ProductionTrainingConfig,
        transactions_path: str | Path | None,
    ) -> ProductionTrainingResult:
        assert Path(data_path) == source
        assert Path(transactions_path) == companion
        assert experiment_name == "fdshield-binary-training"
        assert config.registered_model_name == "fdshield-fraud-detector"
        assert config.auto_promote is True
        assert config.split_datetime == "2026-05-01 00:00:00"
        return ProductionTrainingResult(
            run_id="run-123",
            model_version=17,
            metrics={"validation_pr_auc": 0.9, "validation_recall": 0.85},
            validation_passed=True,
            promoted=True,
        )

    exit_code = main(
        environ,
        stdout=stdout,
        stderr=stderr,
        training_runner=successful_training_runner,
    )
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert [event["event"] for event in events] == [
        "training_job_started",
        "model_registered",
        "training_job_completed",
    ]
    assert events[1]["model_version"] == 17
    assert events[1]["promoted"] is True
