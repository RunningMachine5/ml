from __future__ import annotations

import optuna
import pytest

from fdshield_ml.training.pipeline import TrainingConfig
from fdshield_ml.training.tuning import config_from_best_params, suggest_training_config


@pytest.mark.parametrize(
    ("model_type", "parameters", "expected"),
    [
        (
            "logistic-regression",
            {"logistic_c": 0.1},
            {"logistic_c": 0.1},
        ),
        (
            "decision-tree",
            {"max_depth": 7, "min_samples_leaf": 3},
            {"max_depth": 7, "min_samples_leaf": 3},
        ),
        (
            "random-forest",
            {
                "n_estimators": 250,
                "max_depth": 10,
                "min_samples_leaf": 4,
                "max_features": "log2",
            },
            {
                "n_estimators": 250,
                "max_depth": 10,
                "min_samples_leaf": 4,
                "max_features": "log2",
            },
        ),
        (
            "xgboost",
            {
                "n_estimators": 350,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.75,
                "min_child_weight": 2.0,
            },
            {
                "n_estimators": 350,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.75,
                "min_child_weight": 2.0,
            },
        ),
    ],
)
def test_model_specific_optuna_parameters_update_training_config(
    model_type: str,
    parameters: dict[str, object],
    expected: dict[str, object],
) -> None:
    """각 모델이 자신의 파라미터만 탐색해 공통 TrainingConfig에 반영한다."""

    base = TrainingConfig(model_type=model_type, n_jobs=1)
    trial = optuna.trial.FixedTrial(parameters)

    tuned = suggest_training_config(trial, base)

    for name, value in expected.items():
        assert getattr(tuned, name) == value
    assert tuned.model_type == model_type
    assert tuned.random_state == base.random_state


def test_best_trial_parameters_can_be_restored() -> None:
    base = TrainingConfig(model_type="xgboost", n_jobs=1)
    restored = config_from_best_params(
        base,
        {
            "n_estimators": 450,
            "max_depth": 8,
            "learning_rate": 0.03,
        },
    )

    assert restored.n_estimators == 450
    assert restored.max_depth == 8
    assert restored.learning_rate == 0.03


def test_unknown_best_trial_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unexpected tuning parameters"):
        config_from_best_params(
            TrainingConfig(model_type="xgboost"),
            {"unknown_parameter": 1},
        )
