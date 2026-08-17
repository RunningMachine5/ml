"""train1 raw64 학습·MLflow Registry 등록 흐름 테스트."""

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fdshield_ml.config.preprocess_config import RAW_TRAINING_INPUT_COLUMNS
from fdshield_ml.infrastructure import mlflow as mlflow_integration
from fdshield_ml.infrastructure import training_pipeline
from fdshield_ml.service.train import model_training
from fdshield_ml.service.train.dataset import TRAINING_DATA_CONTRACT
from fdshield_ml.service.train.train_service import prepare_training_data
from fdshield_ml.service.xgboost_prediction import prediction_iteration_range
from tests.conftest import training_row_from_raw51

RawFeaturesFactory = Callable[..., dict[str, object]]


def _training_frame(raw_features_factory: RawFeaturesFactory) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_number in range(40):
        is_fraud = row_number % 2 == 0
        row = training_row_from_raw51(
            raw_features_factory(
                transaction_datetime=f"2026-03-{(row_number % 20) + 1:02d}T10:00:00+09:00",
                transaction_amount=(10_000_000 if is_fraud else 10_000) + row_number,
            ),
            transaction_id=row_number + 1,
            is_fraud=int(is_fraud),
        )
        rows.append(row)
    return pd.DataFrame(rows).loc[:, RAW_TRAINING_INPUT_COLUMNS]


def test_training_accepts_mixed_train1_and_backend_datetime_formats(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    rows: list[dict[str, object]] = []
    for index, (transaction_datetime, is_fraud) in enumerate(
        (("2025-01-01 1:02", 0), ("2025-01-02 01:02:03", 1)),
        start=1,
    ):
        features = raw_features_factory(
            customer_birth_date="1980-01-01",
            customer_registration_datetime=(
                "2012-12-04 4:41" if index == 1 else "2012-12-04 04:41:09"
            ),
            account_creation_datetime=(
                "2017-11-13 0:52" if index == 1 else "2017-11-13 00:52:11"
            ),
            transaction_datetime=transaction_datetime,
            last_atm_transaction_datetime=(
                "2024-12-30 3:04" if index == 1 else "2024-12-30 03:04:05"
            ),
            last_bank_branch_transaction_datetime=None,
            recipient_transaction_resumed_date=(
                "2024-12-01 1:02" if index == 1 else "2024-12-01 01:02:03"
            ),
            account_account_type="e" if index == 2 else "a",
            access_medium=None if index == 2 else "a",
        )
        row = training_row_from_raw51(
            features,
            transaction_id=index,
            is_fraud=is_fraud,
        )
        row["balance_drain_ratio"] = "" if index == 2 else 0.1
        rows.append(row)

    prepared = prepare_training_data(
        pd.DataFrame(rows).loc[:, RAW_TRAINING_INPUT_COLUMNS]
    )

    assert prepared.features.shape == (2, 79)
    assert prepared.features.loc[1, "account_account_type_a"] == 0
    assert prepared.features.loc[
        1,
        "account_remaining_amount_daily_limit_exceeded",
    ] == pytest.approx(6_004_950)


def _install_fake_mlflow(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        mlflow_integration, "configure_tracking", lambda _: "https://mlflow"
    )
    monkeypatch.setattr(mlflow_integration, "verify_connection", lambda: 1)
    monkeypatch.setattr(mlflow_integration.mlflow, "set_experiment", lambda _: None)
    monkeypatch.setattr(
        mlflow_integration.mlflow,
        "log_params",
        lambda params: calls.__setitem__("params", params),
    )
    monkeypatch.setattr(mlflow_integration.mlflow, "log_metrics", lambda _: None)
    monkeypatch.setattr(
        mlflow_integration.mlflow,
        "set_tags",
        lambda tags: calls.__setitem__("run_tags", tags),
    )
    monkeypatch.setattr(
        mlflow_integration.mlflow,
        "log_dict",
        lambda value, path: calls.setdefault("dict_artifacts", {}).__setitem__(
            path, value
        ),
    )

    @contextmanager
    def fake_start_run(*, run_name: str):
        assert run_name == "cloud-run-production-training"
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    monkeypatch.setattr(mlflow_integration.mlflow, "start_run", fake_start_run)

    def fake_log_model(**kwargs: object) -> object:
        calls["model"] = kwargs["sk_model"]
        return SimpleNamespace(registered_model_version=17)

    monkeypatch.setattr(mlflow_integration.mlflow.sklearn, "log_model", fake_log_model)

    class FakeClient:
        def get_model_version_by_alias(self, *args: object) -> object:
            error = RuntimeError("alias not found")
            error.error_code = "RESOURCE_DOES_NOT_EXIST"  # type: ignore[attr-defined]
            raise error

        def set_model_version_tag(self, *args: object) -> None:
            calls.setdefault("tags", []).append(args)

    monkeypatch.setattr(mlflow_integration, "MlflowClient", FakeClient)
    return calls


def test_production_training_config_rejects_invalid_gate() -> None:
    with pytest.raises(ValueError, match="minimum_pr_auc"):
        training_pipeline.ProductionTrainingConfig(
            registered_model_name="fdshield-fraud-detector-v2",
            minimum_pr_auc=1.1,
        )


def test_received_training_defaults_and_manual_review_contract() -> None:
    config = training_pipeline.ProductionTrainingConfig(
        registered_model_name="fdshield-fraud-detector-v2"
    )
    params = model_training.build_classifier(config.model).get_params()

    assert model_training.VALIDATION_FRACTION == pytest.approx(0.2)
    assert model_training.DECISION_THRESHOLD == pytest.approx(0.5)
    assert params["n_estimators"] == 1000
    assert params["learning_rate"] == pytest.approx(0.05)
    assert params["max_depth"] == 6
    assert params["min_child_weight"] == pytest.approx(1.0)
    assert params["gamma"] == pytest.approx(0.0)
    assert params["reg_lambda"] == pytest.approx(1.0)
    assert params["reg_alpha"] == pytest.approx(0.0)
    assert params["subsample"] == pytest.approx(0.8)
    assert params["colsample_bytree"] == pytest.approx(0.8)
    assert params["scale_pos_weight"] == pytest.approx(99.0)
    assert params["tree_method"] == "hist"
    assert params["eval_metric"] == "logloss"
    assert params["early_stopping_rounds"] == 50
    assert params["random_state"] == 42
    assert params["n_jobs"] == -1
    assert (
        training_pipeline.promotion_recommendation(
            {"validation_pr_auc": 0.9},
            None,
            validation_passed=True,
        )
        == "REVIEW_REQUIRED"
    )


def test_legacy_91_feature_champion_is_not_compared() -> None:
    assert (
        mlflow_integration.champion_contract_matches(SimpleNamespace(n_features_in_=91))
        is False
    )
    assert (
        mlflow_integration.champion_contract_matches(SimpleNamespace(n_features_in_=79))
        is True
    )


def test_train1_registers_candidate_with_comparison_metadata(
    tmp_path: Path,
    raw_features_factory: RawFeaturesFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "train1.csv"
    _training_frame(raw_features_factory).to_csv(source, index=False)

    calls = _install_fake_mlflow(monkeypatch)
    prediction_ranges: list[tuple[int, int] | None] = []
    original_predict_proba = model_training.XGBClassifier.predict_proba

    def recording_predict_proba(
        classifier: model_training.XGBClassifier,
        features: object,
        **kwargs: object,
    ) -> object:
        prediction_ranges.append(kwargs.get("iteration_range"))  # type: ignore[arg-type]
        return original_predict_proba(classifier, features, **kwargs)

    monkeypatch.setattr(
        model_training.XGBClassifier,
        "predict_proba",
        recording_predict_proba,
    )

    result = training_pipeline.run_production_training(
        source,
        "fdshield-binary-training",
        training_pipeline.ProductionTrainingConfig(
            registered_model_name="fdshield-fraud-detector-v2",
            minimum_pr_auc=0.0,
            minimum_recall=0.0,
            model=model_training.ModelTrainingConfig(
                n_estimators=2,
                max_depth=2,
                early_stopping_rounds=1,
                n_jobs=1,
            ),
        ),
    )

    assert result.run_id == "run-123"
    assert result.model_version == 17
    assert result.validation_passed is True
    assert result.recommendation == "REVIEW_REQUIRED"
    assert calls["tags"] == [
        ("fdshield-fraud-detector-v2", "17", "validation_status", "passed"),
        (
            "fdshield-fraud-detector-v2",
            "17",
            "promotion_recommendation",
            "REVIEW_REQUIRED",
        ),
        ("fdshield-fraud-detector-v2", "17", "decision_threshold", "0.5"),
        (
            "fdshield-fraud-detector-v2",
            "17",
            "feature_contract",
            TRAINING_DATA_CONTRACT,
        ),
    ]
    assert calls["params"]["training_data_contract"] == "train1-raw64"
    assert calls["params"]["feature_contract"] == TRAINING_DATA_CONTRACT
    assert calls["params"]["split_strategy"] == "random_stratified_80_20"
    assert calls["params"]["validation_fraction"] == pytest.approx(0.2)
    assert calls["params"]["decision_threshold"] == pytest.approx(0.5)
    assert calls["model"].decision_threshold_ == pytest.approx(0.5)
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
    assert comparison["champion_comparison_status"] == "not_available"
    assert comparison["recommendation"] == result.recommendation
    feature_schema = calls["dict_artifacts"]["metadata/model-feature-schema.json"]
    assert feature_schema["feature_contract"] == TRAINING_DATA_CONTRACT
    assert feature_schema["feature_count"] == 79
    assert calls["run_tags"]["feature_contract"] == TRAINING_DATA_CONTRACT
    assert calls["run_tags"]["champion_comparison_status"] == "not_available"


def test_champion_evaluation_uses_registered_model_threshold_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model79Champion:
        n_features_in_ = 79

        def predict_proba(self, features: pd.DataFrame) -> object:
            assert len(features) == 2
            return np.asarray([[0.4, 0.6], [0.3, 0.7]])

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> object:
            assert name == "fdshield-fraud-detector-v2"
            assert alias == "champion"
            return SimpleNamespace(version="5")

        def get_model_version(self, name: str, version: str) -> object:
            assert name == "fdshield-fraud-detector-v2"
            assert version == "5"
            return SimpleNamespace(tags={"decision_threshold": "0.65"})

    monkeypatch.setattr(
        mlflow_integration.mlflow.sklearn,
        "load_model",
        lambda _: Model79Champion(),
    )

    evaluation = mlflow_integration.evaluate_champion(
        FakeClient(),  # type: ignore[arg-type]
        registered_model_name="fdshield-fraud-detector-v2",
        model_alias="champion",
        features=pd.DataFrame(np.zeros((2, 79))),
        target=pd.Series([0, 1]),
    )

    assert evaluation.model_version == 5
    assert evaluation.metrics is not None
    assert evaluation.metrics["decision_threshold"] == pytest.approx(0.65)
    assert evaluation.metrics["validation_recall"] == pytest.approx(1.0)


def test_legacy_91_feature_champion_comparison_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyChampion:
        n_features_in_ = 91

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> object:
            assert name == "fdshield-fraud-detector-v2"
            assert alias == "champion"
            return SimpleNamespace(version="5")

        def get_model_version(self, name: str, version: str) -> object:
            assert name == "fdshield-fraud-detector-v2"
            assert version == "5"
            return SimpleNamespace(tags={"decision_threshold": "0.55"})

    monkeypatch.setattr(
        mlflow_integration.mlflow.sklearn,
        "load_model",
        lambda _: LegacyChampion(),
    )

    evaluation = mlflow_integration.evaluate_champion(
        FakeClient(),  # type: ignore[arg-type]
        registered_model_name="fdshield-fraud-detector-v2",
        model_alias="champion",
        features=pd.DataFrame(np.zeros((2, 79))),
        target=pd.Series([0, 1]),
    )

    assert evaluation.model_version == 5
    assert evaluation.metrics is None


def test_recommendation_requires_relative_improvement_without_guardrail_regression() -> (
    None
):
    champion = {
        "validation_pr_auc": 0.90,
        "validation_recall": 0.85,
        "validation_fpr": 0.01,
    }

    assert (
        training_pipeline.promotion_recommendation(
            {
                "validation_pr_auc": 0.91,
                "validation_recall": 0.86,
                "validation_fpr": 0.009,
            },
            champion,
            validation_passed=True,
        )
        == "RECOMMENDED"
    )
    assert (
        training_pipeline.promotion_recommendation(
            {
                "validation_pr_auc": 0.91,
                "validation_recall": 0.80,
                "validation_fpr": 0.009,
            },
            champion,
            validation_passed=True,
        )
        == "REVIEW_REQUIRED"
    )
    assert (
        training_pipeline.promotion_recommendation(
            {
                "validation_pr_auc": 0.89,
                "validation_recall": 0.90,
                "validation_fpr": 0.009,
            },
            champion,
            validation_passed=True,
        )
        == "NOT_RECOMMENDED"
    )
