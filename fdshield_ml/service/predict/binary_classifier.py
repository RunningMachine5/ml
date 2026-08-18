"""
[사기 예측 코드]
전처리가 끝난 model79 한 건을 모델에 넣어 예측한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from fdshield_ml.service.xgboost_prediction import prediction_iteration_range


def predict(model: object, data: pd.DataFrame) -> dict[str, object]:
    """사기 여부, 사기 확률, model79 기여도를 계산한다."""

    # 예측과 확률 계산
    predict_result = model.predict(data)
    predict_proba = model.predict_proba(data)

    # XGBoost 모델이면 각 피처의 기여도도 계산한다.
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
        shap_values = {
            column: float(value)
            for column, value in zip(data.columns, contributions[0, :-1])
        }

    return {
        "predict_result": int(predict_result[0]),
        "predict_proba": float(predict_proba[0, 1]),
        "shap_values": shap_values,
    }
