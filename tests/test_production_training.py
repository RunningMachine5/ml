"""실제 학습·MLflow Registry 등록 흐름 테스트."""

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fdshield_ml.common.preprocessing import preprocess_frame
from fdshield_ml.common.xgboost_prediction import prediction_iteration_range
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
    monkeypatch.setattr(
        production.mlflow,
        "set_tags",
        lambda tags: calls.__setitem__("run_tags", tags),
    )
    monkeypatch.setattr(
        production.mlflow,
        "log_dict",
        lambda value, path: calls.setdefault("dict_artifacts", {}).__setitem__(
            path, value
        ),
    )
    monkeypatch.setattr(
        production.mlflow,
        "log_text",
        lambda *_: pytest.fail("Production training must not log text artifacts"),
    )

    @contextmanager
    def fake_start_run(*, run_name: str):
        assert run_name == "cloud-run-production-training"
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    monkeypatch.setattr(production.mlflow, "start_run", fake_start_run)

    def fake_log_model(**kwargs: object) -> object:
        calls["model"] = kwargs["sk_model"]
        return SimpleNamespace(registered_model_version=17)

    monkeypatch.setattr(production.mlflow.sklearn, "log_model", fake_log_model)

    class FakeClient:
        def get_model_version_by_alias(self, *args: object) -> object:
            error = RuntimeError("alias not found")
            error.error_code = "RESOURCE_DOES_NOT_EXIST"  # type: ignore[attr-defined]
            raise error

        def set_model_version_tag(self, *args: object) -> None:
            calls.setdefault("tags", []).append(args)

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


def test_train_registers_candidate_with_comparison_metadata(
    tmp_path: Path,
    raw_features_factory: RawFeaturesFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "transactions.csv"
    _training_frame(raw_features_factory).to_csv(source, index=False)

    calls = _install_fake_mlflow(monkeypatch)
    prediction_ranges: list[tuple[int, int] | None] = []
    original_predict_proba = production.XGBClassifier.predict_proba

    def recording_predict_proba(
        classifier: production.XGBClassifier,
        features: object,
        **kwargs: object,
    ) -> object:
        prediction_ranges.append(kwargs.get("iteration_range"))  # type: ignore[arg-type]
        return original_predict_proba(classifier, features, **kwargs)

    monkeypatch.setattr(
        production.XGBClassifier,
        "predict_proba",
        recording_predict_proba,
    )

    result = production.train_and_register_model(
        source,
        "fdshield-binary-training",
        production.ProductionTrainingConfig(
            registered_model_name="fdshield-fraud-detector",
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
    assert result.recommendation == "REVIEW_REQUIRED"
    assert calls["tags"][:2] == [
        ("fdshield-fraud-detector", "17", "validation_status", "passed"),
        (
            "fdshield-fraud-detector",
            "17",
            "promotion_recommendation",
            "REVIEW_REQUIRED",
        ),
    ]
    assert calls["tags"][2] == (
        "fdshield-fraud-detector",
        "17",
        "decision_threshold",
        repr(result.metrics["decision_threshold"]),
    )
    assert "alias" not in calls
    assert calls["params"]["training_data_contract"] == "raw"
    assert calls["params"]["split_strategy"] == "time"
    assert calls["params"]["decision_threshold"] == result.metrics["decision_threshold"]
    assert calls["model"].decision_threshold_ == result.metrics["decision_threshold"]
    expected_range = prediction_iteration_range(
        calls["model"],
        calls["model"].get_booster(),
    )
    assert prediction_ranges == [expected_range, expected_range]
    assert set(calls["dict_artifacts"]) == {
        "metadata/model-comparison.json",
        "metadata/model-feature-schema.json",
    }
    comparison = calls["dict_artifacts"]["metadata/model-comparison.json"]
    assert comparison["candidate"] == result.metrics
    assert comparison["champion"] is None
    assert comparison["recommendation"] == result.recommendation
    assert calls["run_tags"] == {
        "project": "fdshield",
        "task": "binary_fraud_detection",
        "pipeline_stage": "production_training",
        "validation_status": "passed",
        "promotion_recommendation": "REVIEW_REQUIRED",
        "champion_model_version": "",
    }
    assert len(calls["tags"]) == 3


def test_champion_evaluation_uses_registered_model_threshold_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyChampion:
        def predict_proba(self, features: pd.DataFrame) -> object:
            assert len(features) == 2
            return np.asarray([[0.4, 0.6], [0.3, 0.7]])

    class FakeClient:
        def get_model_version(self, name: str, version: str) -> object:
            assert name == "fdshield-fraud-detector"
            assert version == "5"
            return SimpleNamespace(tags={"decision_threshold": "0.65"})

    monkeypatch.setattr(
        production.mlflow.sklearn,
        "load_model",
        lambda _: LegacyChampion(),
    )

    version, metrics = production._champion_evaluation(
        FakeClient(),  # type: ignore[arg-type]
        production.ProductionTrainingConfig(
            registered_model_name="fdshield-fraud-detector",
            champion_model_version=5,
        ),
        pd.DataFrame({"feature": [0.0, 1.0]}),
        pd.Series([0, 1]),
    )

    assert version == 5
    assert metrics is not None
    assert metrics["decision_threshold"] == pytest.approx(0.65)
    assert metrics["validation_recall"] == pytest.approx(1.0)


def test_recommendation_requires_relative_improvement_without_guardrail_regression() -> None:
    champion = {
        "validation_pr_auc": 0.90,
        "validation_recall": 0.85,
        "validation_fpr": 0.01,
    }

    assert production._promotion_recommendation(
        {
            "validation_pr_auc": 0.91,
            "validation_recall": 0.86,
            "validation_fpr": 0.009,
        },
        champion,
        validation_passed=True,
    ) == "RECOMMENDED"
    assert production._promotion_recommendation(
        {
            "validation_pr_auc": 0.91,
            "validation_recall": 0.80,
            "validation_fpr": 0.009,
        },
        champion,
        validation_passed=True,
    ) == "REVIEW_REQUIRED"
    assert production._promotion_recommendation(
        {
            "validation_pr_auc": 0.89,
            "validation_recall": 0.90,
            "validation_fpr": 0.009,
        },
        champion,
        validation_passed=True,
    ) == "NOT_RECOMMENDED"


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
