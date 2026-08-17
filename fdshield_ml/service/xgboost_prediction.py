"""XGBoost 확률과 SHAP 계산이 같은 트리 범위를 사용하도록 돕는다."""

from __future__ import annotations


def prediction_iteration_range(model: object, booster: object) -> tuple[int, int]:
    """scikit-learn wrapper가 예측에 사용하는 ``[begin, end)`` 범위를 반환한다."""

    try:
        best_iteration = model.best_iteration
    except (AttributeError, ValueError):
        best_iteration = None

    if best_iteration is not None:
        try:
            end = int(best_iteration) + 1
        except (TypeError, ValueError) as exc:
            raise ValueError("XGBoost best_iteration must be an integer") from exc
        if end < 1:
            raise ValueError("XGBoost best_iteration must not be negative")
        return (0, end)

    num_boosted_rounds = getattr(booster, "num_boosted_rounds", None)
    if num_boosted_rounds is None:
        raise TypeError("XGBoost booster must provide num_boosted_rounds()")
    rounds = int(num_boosted_rounds())
    if rounds < 1:
        raise ValueError("XGBoost booster must contain at least one tree round")
    return (0, rounds)
