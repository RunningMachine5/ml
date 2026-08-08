"""실제 학습·MLflow Registry 등록 흐름 테스트."""

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fdshield_ml.training import production


RawFeaturesFactory = Callable[..., dict[str, object]]


def _training_frame(raw_features_factory: RawFeaturesFactory) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_number in range(10):
        for row_number in range(4):
            is_fraud = row_number == 0
            rows.append(
                {
                    **raw_features_factory(
                        Transaction_Amount=(10_000_000 if is_fraud else 10_000)
                        + group_number,
                    ),
                    "Account_account_number": f"account-{group_number}",
                    "Is_Fraud": int(is_fraud),
                }
            )
    return pd.DataFrame(rows)


def test_production_training_config_rejects_invalid_gate() -> None:
    with pytest.raises(ValueError, match="minimum_pr_auc"):
        production.ProductionTrainingConfig(
            registered_model_name="fdshield-fraud-detector",
            minimum_pr_auc=1.1,
        )


def test_train_registers_version_and_promotes_passing_model(
    tmp_path: Path,
    raw_features_factory: RawFeaturesFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "transactions.csv"
    _training_frame(raw_features_factory).to_csv(source, index=False)

    monkeypatch.setattr(production, "configure_tracking", lambda _: "https://mlflow")
    monkeypatch.setattr(production, "verify_connection", lambda: 1)
    monkeypatch.setattr(production.mlflow, "set_experiment", lambda _: None)
    monkeypatch.setattr(production.mlflow, "log_params", lambda _: None)
    monkeypatch.setattr(production.mlflow, "log_metrics", lambda _: None)
    monkeypatch.setattr(production.mlflow, "set_tags", lambda _: None)
    monkeypatch.setattr(production.mlflow, "log_dict", lambda *_: None)

    @contextmanager
    def fake_start_run(*, run_name: str):
        assert run_name == "cloud-run-production-training"
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    monkeypatch.setattr(production.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(
        production.mlflow.sklearn,
        "log_model",
        lambda **_: SimpleNamespace(registered_model_version=17),
    )

    calls: dict[str, tuple[object, ...]] = {}

    class FakeClient:
        def set_model_version_tag(self, *args: object) -> None:
            calls["tag"] = args

        def set_registered_model_alias(self, *args: object) -> None:
            calls["alias"] = args

    monkeypatch.setattr(production, "MlflowClient", FakeClient)

    result = production.train_and_register_model(
        source,
        "fdshield-binary-training",
        production.ProductionTrainingConfig(
            registered_model_name="fdshield-fraud-detector",
            auto_promote=True,
            minimum_pr_auc=0.0,
            minimum_recall=0.0,
            n_estimators=2,
            max_depth=2,
            n_jobs=1,
        ),
    )

    assert result.run_id == "run-123"
    assert result.model_version == 17
    assert result.validation_passed is True
    assert result.promoted is True
    assert calls["tag"] == (
        "fdshield-fraud-detector",
        "17",
        "validation_status",
        "passed",
    )
    assert calls["alias"] == (
        "fdshield-fraud-detector",
        "champion",
        "17",
    )
