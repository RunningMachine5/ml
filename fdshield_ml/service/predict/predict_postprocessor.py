"""XGBoost model80 기여도를 API용 원본 Feature 그룹으로 후처리한다."""

from fdshield_ml.config.preprocess_config import (
    CATEGORICAL_LEVELS,
    MODEL_FEATURE_COLUMNS,
)

_SHAP_SCALAR_OUTPUT_NAMES = tuple(
    {
        "seconds_since_last_transaction": "time_difference",
        "distance_since_last_transaction": "distance",
    }.get(name, name)
    for name in MODEL_FEATURE_COLUMNS[:49]
)


def shap_decode(shap: dict[str, float]) -> dict[str, float]:
    """model80 기여도를 전달받은 ML API의 원본/파생 56개 그룹으로 합친다."""

    if not shap:
        return {}
    if set(shap) != set(MODEL_FEATURE_COLUMNS):
        raise ValueError("SHAP values must contain the exact model80 feature set")
    decoded = {
        output_name: float(shap[model_name])
        for model_name, output_name in zip(
            MODEL_FEATURE_COLUMNS[:49],
            _SHAP_SCALAR_OUTPUT_NAMES,
            strict=True,
        )
    }
    for field, levels in CATEGORICAL_LEVELS.items():
        decoded[field] = float(sum(shap[f"{field}_{level}"] for level in levels))
    return decoded


__all__ = ["shap_decode"]
