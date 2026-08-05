"""Cloud Run Training Job 진입점 테스트."""

from __future__ import annotations

import io
import json

import pytest

from fdshield_ml.training.job import TrainingJobConfig, main


VALID_ENV = {
    "TRAINING_JOB_TYPE": "binary",
    "TRAINING_DATA_URI": "stub://local-data",
    "MLFLOW_EXPERIMENT_NAME": "fdshield-binary-training",
}


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
