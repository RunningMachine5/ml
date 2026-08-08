"""실제 학습·MLflow Registry 등록 흐름 테스트."""

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fdshield_ml.common.preprocessing import preprocess_frame
from fdshield_ml.training import production


RawFeaturesFactory = Callable[..., dict[str, object]]


def _training_frame(raw_features_factory: RawFeaturesFactory) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_number in range(10):
        for row_number in range(4):
            is_fraud = row_number % 2 == 0
            rows.append(
                {
                    **raw_features_factory(
                        Transaction_Datetime=(
                            "2026-03-30 10:00:00"
                            if row_number < 2
                            else "2026-04-02 10:00:00"
                        ),
                        Transaction_Amount=(10_000_000 if is_fraud else 10_000)
                        + group_number,
                    ),
                    "Account_account_number": f"account-{group_number}",
                    "Is_Fraud": int(is_fraud),
                }
            )
    return pd.DataFrame(rows)


def _install_fake_mlflow(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    calls: dict[str, object] = {}
    monkeypatch.setattr(production, "configure_tracking", lambda _: "https://mlflow")
    monkeypatch.setattr(production, "verify_connection", lambda: 1)
    monkeypatch.setattr(production.mlflow, "set_experiment", lambda _: None)
    monkeypatch.setattr(
        production.mlflow,
        "log_params",
        lambda params: calls.__setitem__("params", params),
    )
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

    class FakeClient:
        def set_model_version_tag(self, *args: object) -> None:
            calls["tag"] = args

        def set_registered_model_alias(self, *args: object) -> None:
            calls["alias"] = args

    monkeypatch.setattr(production, "MlflowClient", FakeClient)
    return calls


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

    calls = _install_fake_mlflow(monkeypatch)

    result = production.train_and_register_model(
        source,
        "fdshield-binary-training",
        production.ProductionTrainingConfig(
            registered_model_name="fdshield-fraud-detector",
            auto_promote=True,
            minimum_pr_auc=0.0,
            minimum_recall=0.0,
            minimum_fraud_rows_per_split=1,
            n_estimators=2,
            max_depth=2,
            early_stopping_rounds=1,
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
    assert calls["params"]["training_data_contract"] == "raw"
    assert calls["params"]["split_strategy"] == "time"


def test_train_accepts_exact_preprocessed_contract_with_time_split(
    tmp_path: Path,
    raw_features_factory: RawFeaturesFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_rows: list[dict[str, object]] = []
    labels: list[int] = []
    datetimes: list[str] = []
    for side, transaction_datetime in enumerate(
        ("2026-03-30 10:00:00", "2026-04-02 10:00:00")
    ):
        for row_number in range(10):
            is_fraud = row_number % 2
            raw_rows.append(
                raw_features_factory(
                    Transaction_Datetime=transaction_datetime,
                    Transaction_Amount=10_000_000 if is_fraud else 10_000,
                )
            )
            labels.append(is_fraud)
            datetimes.append(transaction_datetime)

    preprocessed = preprocess_frame(pd.DataFrame(raw_rows))
    preprocessed["Is_Fraud"] = labels
    training_path = tmp_path / "train.csv"
    transactions_path = tmp_path / "transactions.csv"
    preprocessed.to_csv(training_path, index=False)
    pd.DataFrame(
        {
            "Transaction_Datetime": datetimes,
            "Is_Fraud": labels,
        }
    ).to_csv(transactions_path, index=False)
    calls = _install_fake_mlflow(monkeypatch)

    result = production.train_and_register_model(
        training_path,
        "fdshield-binary-training",
        production.ProductionTrainingConfig(
            registered_model_name="fdshield-fraud-detector",
            minimum_fraud_rows_per_split=1,
            n_estimators=2,
            max_depth=2,
            early_stopping_rounds=1,
            n_jobs=1,
        ),
        transactions_path,
    )

    assert result.model_version == 17
    assert calls["params"]["training_data_contract"] == "preprocessed"
    assert calls["params"]["split_strategy"] == "time"
    assert calls["params"]["split_datetime"] == "2026-04-01 00:00:00"


def test_preprocessed_training_requires_companion_transactions(
    tmp_path: Path,
    raw_features_factory: RawFeaturesFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [raw_features_factory() for _ in range(10)]
    source = preprocess_frame(pd.DataFrame(rows))
    source["Is_Fraud"] = [0, 1] * 5
    path = tmp_path / "train.csv"
    source.to_csv(path, index=False)
    monkeypatch.setattr(production, "configure_tracking", lambda _: "https://mlflow")
    monkeypatch.setattr(production, "verify_connection", lambda: 1)

    with pytest.raises(
        production.ProductionTrainingError,
        match="requires companion transactions",
    ):
        production.train_and_register_model(
            path,
            "fdshield-binary-training",
            production.ProductionTrainingConfig(
                registered_model_name="fdshield-fraud-detector",
            ),
        )
