"""학습과 추론이 공유하는 사기 판정 임계값 계약."""

from __future__ import annotations

import math
from collections.abc import Mapping

DECISION_THRESHOLD_ATTRIBUTE = "decision_threshold_"
DECISION_THRESHOLD_TAG = "decision_threshold"


class DecisionThresholdError(ValueError):
    """모델의 판정 임계값이 없거나 운영 계약을 만족하지 않을 때 발생한다."""


def validate_decision_threshold(value: object, *, source: str) -> float:
    """임계값을 유한한 0~1 실수로 정규화한다."""

    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionThresholdError(
            f"{source} must be a number between 0 and 1"
        ) from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise DecisionThresholdError(f"{source} must be a number between 0 and 1")
    return threshold


def store_model_decision_threshold(model: object, value: object) -> float:
    """직렬화되는 모델 객체에 판정 임계값을 함께 저장한다."""

    threshold = validate_decision_threshold(value, source="decision threshold")
    setattr(model, DECISION_THRESHOLD_ATTRIBUTE, threshold)
    return threshold


def resolve_model_decision_threshold(
    model: object,
    *,
    model_version_tags: Mapping[str, str] | None = None,
) -> float:
    """모델 속성을 우선하고 기존 모델은 Registry 태그에서 임계값을 읽는다."""

    stored_value = getattr(model, DECISION_THRESHOLD_ATTRIBUTE, None)
    tagged_value = (
        model_version_tags.get(DECISION_THRESHOLD_TAG)
        if model_version_tags is not None
        else None
    )
    if stored_value is None and tagged_value is None:
        raise DecisionThresholdError(
            "registered model does not contain a decision threshold"
        )

    stored_threshold = (
        validate_decision_threshold(stored_value, source="model decision threshold")
        if stored_value is not None
        else None
    )
    tagged_threshold = (
        validate_decision_threshold(tagged_value, source="model version threshold tag")
        if tagged_value is not None
        else None
    )
    if (
        stored_threshold is not None
        and tagged_threshold is not None
        and not math.isclose(stored_threshold, tagged_threshold, abs_tol=1e-12)
    ):
        raise DecisionThresholdError(
            "model decision threshold and model version tag do not match"
        )
    if stored_threshold is not None:
        return stored_threshold
    assert tagged_threshold is not None
    return tagged_threshold


__all__ = [
    "DECISION_THRESHOLD_ATTRIBUTE",
    "DECISION_THRESHOLD_TAG",
    "DecisionThresholdError",
    "resolve_model_decision_threshold",
    "store_model_decision_threshold",
    "validate_decision_threshold",
]
