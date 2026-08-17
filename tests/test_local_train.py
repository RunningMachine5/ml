"""MLflow 없이 사용하는 로컬 학습 명령 테스트."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fdshield_ml import local_train


class FakeModel:
    def save_model(self, path: str) -> None:
        Path(path).write_text('{"model":"fake-model79"}', encoding="utf-8")


def _fake_training_result() -> SimpleNamespace:
    return SimpleNamespace(
        model=FakeModel(),
        metrics={"validation_pr_auc": 0.81, "validation_recall": 0.72},
        train_rows=160,
        validation_rows=40,
        iteration_range=(0, 7),
        decision_threshold=0.5,
    )


def test_train_local_bundle_saves_and_validates_serving_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated: list[Path] = []
    monkeypatch.setattr(
        local_train,
        "ml_train_flow",
        lambda _path, _config: _fake_training_result(),
    )
    monkeypatch.setattr(
        local_train,
        "load_local_predict_service",
        lambda path: validated.append(path),
    )
    output_dir = tmp_path / "bundle"

    summary = local_train.train_local_bundle(
        data_path=tmp_path / "train1.csv",
        output_dir=output_dir,
        model_name="fdshield-fraud-detector-v2",
        model_version="17",
    )

    model_path = output_dir / "model.json"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert validated == [output_dir.resolve()]
    assert manifest == {
        "bundle_schema_version": 1,
        "decision_threshold": 0.5,
        "feature_count": 79,
        "model_file": "model.json",
        "model_format": "xgboost-json",
        "model_name": "fdshield-fraud-detector-v2",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_version": "17",
    }
    assert metrics == {
        "iteration_range": [0, 7],
        "metrics": {"validation_pr_auc": 0.81, "validation_recall": 0.72},
        "train_rows": 160,
        "validation_rows": 40,
    }
    assert summary["bundle_path"] == str(output_dir.resolve())
    assert summary["model_version"] == "17"


def test_train_local_bundle_rejects_nonempty_output_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
    called = False

    def fake_train(_path: Path, _config: object) -> SimpleNamespace:
        nonlocal called
        called = True
        return _fake_training_result()

    monkeypatch.setattr(local_train, "ml_train_flow", fake_train)

    with pytest.raises(FileExistsError, match="not empty"):
        local_train.train_local_bundle(
            data_path=tmp_path / "train1.csv",
            output_dir=output_dir,
            model_name="fdshield-fraud-detector-v2",
            model_version="1",
        )

    assert called is False
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_local_train_main_reports_invalid_model_version() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = local_train.main(
        ["--model-version", "latest"],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "positive integer" in stderr.getvalue()
