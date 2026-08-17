"""전처리가 끝난 model79 한 건의 확률과 SHAP 기여도를 계산한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from fdshield_ml.service.xgboost_prediction import prediction_iteration_range


class BinaryClassifierError(ValueError):
    """모델 확률 또는 SHAP 결과가 Serving 계약을 만족하지 않을 때 발생한다."""


def predict(model: object, data: pd.DataFrame) -> dict[str, object]:
    """전처리된 model79 한 건의 확률과 model79 SHAP 기여도를 계산한다."""

    probabilities = np.asarray(model.predict_proba(data), dtype="float64")
    if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
        raise BinaryClassifierError(
            "predict_proba must return one finite binary row; "
            f"shape={probabilities.shape}"
        )
    shap_values: dict[str, float] = {}
    get_booster = getattr(model, "get_booster", None)
    if get_booster is not None:
        booster = get_booster()
        matrix = xgb.DMatrix(data, feature_names=list(data.columns))
        contributions = np.asarray(
            booster.predict(
                matrix,
                pred_contribs=True,
                iteration_range=prediction_iteration_range(model, booster),
            ),
            dtype="float64",
        )
        if contributions.shape != (1, len(data.columns) + 1):
            raise BinaryClassifierError(
                f"Unexpected contribution shape: {contributions.shape}"
            )
        shap_values = {
            column: float(value)
            for column, value in zip(
                data.columns,
                contributions[0, :-1],
                strict=True,
            )
        }
    return {
        "predict_proba": float(probabilities[0, 1]),
        "shap_values": shap_values,
    }


__all__ = [
    "BinaryClassifierError",
    "predict",
]
