"""XGBoost 확률과 SHAP이 공유하는 트리 범위 계약 테스트."""

from __future__ import annotations

import pytest

from fdshield_ml.common.xgboost_prediction import prediction_iteration_range


class _Booster:
    def __init__(self, rounds: int = 8) -> None:
        self.rounds = rounds

    def num_boosted_rounds(self) -> int:
        return self.rounds


def test_prediction_range_stops_after_best_iteration() -> None:
    model = type("EarlyStoppedModel", (), {"best_iteration": 2})()

    assert prediction_iteration_range(model, _Booster()) == (0, 3)


def test_prediction_range_uses_all_rounds_without_early_stopping() -> None:
    assert prediction_iteration_range(object(), _Booster(11)) == (0, 11)


def test_prediction_range_rejects_empty_booster() -> None:
    with pytest.raises(ValueError, match="at least one tree"):
        prediction_iteration_range(object(), _Booster(0))
