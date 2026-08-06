"""Training Job의 MLflow 검증 Run 기록 테스트."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fdshield_ml.training.data_loader import TrainingDataSummary
from fdshield_ml.training import job_tracking


class FakeRun:
    """MLflow start_run 컨텍스트의 테스트 대역."""

    info = SimpleNamespace(run_id="run-validation-123")

    def __enter__(self) -> "FakeRun":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_log_data_validation_run_records_summary_without_raw_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        job_tracking,
        "configure_tracking",
        lambda env_file: calls.setdefault("env_file", env_file),
    )
    monkeypatch.setattr(
        job_tracking,
        "verify_connection",
        lambda: calls.setdefault("connection_verified", True),
    )
    monkeypatch.setattr(
        job_tracking.mlflow,
        "set_experiment",
        lambda name: calls.setdefault("experiment_name", name),
    )
    monkeypatch.setattr(
        job_tracking.mlflow,
        "start_run",
        lambda run_name: calls.setdefault("run_name", run_name) and FakeRun(),
    )
    monkeypatch.setattr(
        job_tracking.mlflow,
        "log_params",
        lambda values: calls.setdefault("params", values),
    )
    monkeypatch.setattr(
        job_tracking.mlflow,
        "log_metrics",
        lambda values: calls.setdefault("metrics", values),
    )
    monkeypatch.setattr(
        job_tracking.mlflow,
        "set_tags",
        lambda values: calls.setdefault("tags", values),
    )
    monkeypatch.setattr(
        job_tracking.mlflow,
        "log_dict",
        lambda values, path: calls.update({"artifact": values, "artifact_path": path}),
    )
    summary = TrainingDataSummary(
        row_count=120_000,
        column_count=64,
        normal_count=118_800,
        fraud_count=1_200,
        file_size_bytes=56_025_173,
    )

    run_id = job_tracking.log_data_validation_run(
        "binary",
        "gs://fdshield-ml-data-801817539291/datasets/open/v1/train.csv",
        "fdshield-binary-training",
        "gcs",
        summary,
    )

    assert run_id == "run-validation-123"
    assert calls["env_file"] is None
    assert calls["connection_verified"] is True
    assert calls["experiment_name"] == "fdshield-binary-training"
    assert calls["run_name"] == "binary-data-validation"
    assert calls["metrics"] == {
        "row_count": 120_000,
        "column_count": 64,
        "normal_count": 118_800,
        "fraud_count": 1_200,
        "file_size_bytes": 56_025_173,
    }
    assert "raw_rows" not in calls["artifact"]
    assert calls["artifact_path"] == "validation/training-data-summary.json"


def test_log_data_validation_run_wraps_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_configure(env_file: object) -> str:
        raise ValueError("invalid credentials")

    monkeypatch.setattr(job_tracking, "configure_tracking", fail_to_configure)

    with pytest.raises(
        job_tracking.TrainingTrackingError,
        match="Failed to record",
    ):
        job_tracking.log_data_validation_run(
            "binary",
            "data/open/train.csv",
            "fdshield-binary-training",
            "local",
            TrainingDataSummary(2, 64, 1, 1, 1024),
        )
