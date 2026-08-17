"""XGBoost model79 기여도를 API용 원본 Feature 그룹으로 후처리한다."""

from fdshield_ml.config.preprocess_config import (
    CATEGORICAL_LEVELS,
    MODEL_FEATURE_COLUMNS,
)

_ENCODED_FEATURE_NAMES = frozenset(
    f"{field}_{level}"
    for field, levels in CATEGORICAL_LEVELS.items()
    for level in levels
)
_SHAP_SCALAR_FEATURE_NAMES = tuple(
    name for name in MODEL_FEATURE_COLUMNS if name not in _ENCODED_FEATURE_NAMES
)
_SHAP_SCALAR_OUTPUT_NAMES = tuple(
    {
        "seconds_since_last_transaction": "time_difference",
        "distance_since_last_transaction": "distance",
    }.get(name, name)
    for name in _SHAP_SCALAR_FEATURE_NAMES
)


def shap_decode(shap: dict[str, float]) -> dict[str, float]:
    """model79 기여도를 원본·파생·범주 그룹으로 합친다."""

    if not shap:
        return {}
    if set(shap) != set(MODEL_FEATURE_COLUMNS):
        raise ValueError("SHAP values must contain the exact model79 feature set")
    decoded = {
        output_name: float(shap[model_name])
        for model_name, output_name in zip(
            _SHAP_SCALAR_FEATURE_NAMES,
            _SHAP_SCALAR_OUTPUT_NAMES,
            strict=True,
        )
    }
    for field, levels in CATEGORICAL_LEVELS.items():
        decoded[field] = float(sum(shap[f"{field}_{level}"] for level in levels))
    return decoded
