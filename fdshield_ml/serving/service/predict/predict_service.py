"""전처리, 이진 분류, SHAP 후처리를 잇는 ML 예측 흐름."""

from __future__ import annotations

from fdshield_ml.common.decision_threshold import (
    DecisionThresholdError,
    resolve_model_decision_threshold,
)
from fdshield_ml.common.preprocessor import Preprocessor
from fdshield_ml.serving.dto.predict_input import PredictInputDTO
from fdshield_ml.serving.dto.predict_result import PredictResultDTO
from fdshield_ml.serving.service.predict.binary_classifier import (
    BinaryClassifierError,
    predict,
)
from fdshield_ml.serving.service.predict.predict_postprocessor import shap_decode


class PredictionServiceError(RuntimeError):
    """모델 로딩 또는 예측 결과가 Serving 계약을 만족하지 않을 때 발생한다."""


class PredictService:
    """raw60을 model80으로 바꾸고 사기 확률과 설명값을 반환한다."""

    def __init__(
        self,
        *,
        model: object,
        model_name: str,
        model_version: str,
        model_version_tags: dict[str, str] | None = None,
    ) -> None:
        if not model_name.strip() or not model_version.strip():
            raise ValueError("model_name and model_version are required")
        if not hasattr(model, "predict_proba"):
            raise TypeError("The model must provide predict_proba().")
        try:
            threshold = resolve_model_decision_threshold(
                model,
                model_version_tags=model_version_tags,
            )
        except DecisionThresholdError as exc:
            raise PredictionServiceError(str(exc)) from exc
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.threshold = threshold
        self.ready = True
        self.preprocessor = Preprocessor()

    def predict(self, request: PredictInputDTO) -> PredictResultDTO:
        """거래 한 건을 전처리해 예측·기여도를 반환한다."""

        features = self.preprocessor.predict_preprocess(request)
        try:
            prediction = predict(self.model, features)
        except BinaryClassifierError as exc:
            raise PredictionServiceError(str(exc)) from exc
        except Exception as exc:
            raise PredictionServiceError("Model prediction failed") from exc
        fraud_probability = float(prediction["predict_proba"])
        return PredictResultDTO(
            transaction_id=request.transaction_id,
            predict_result=int(fraud_probability >= self.threshold),
            predict_proba=fraud_probability,
            shap_values=shap_decode(prediction["shap_values"]),
            model_name=self.model_name,
            model_version=self.model_version,
        )


def ml_predict_flow(
    transaction: PredictInputDTO,
    service: PredictService,
) -> PredictResultDTO:
    """전달본과 같은 이름으로 한 건의 전체 예측 흐름을 실행한다."""

    return service.predict(transaction)


__all__ = [
    "PredictService",
    "PredictionServiceError",
    "ml_predict_flow",
]
