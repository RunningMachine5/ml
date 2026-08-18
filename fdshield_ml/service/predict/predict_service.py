"""
[ML 예측 서비스]
컨트롤러 -> 전처리 -> 추론 -> 후처리 흐름을 담당한다.
"""

from __future__ import annotations

from fdshield_ml.dto.predict_input import PredictInputDTO
from fdshield_ml.dto.predict_result import PredictResultDTO
from fdshield_ml.service.predict.binary_classifier import predict
from fdshield_ml.service.predict.predict_postprocessor import shap_decode
from fdshield_ml.service.preprocessor import Preprocessor


class PredictionServiceError(RuntimeError):
    """모델 파일을 불러오지 못했을 때 발생한다."""


class PredictService:
    """raw51을 model79로 바꾸고 사기 확률과 설명값을 반환한다."""

    def __init__(
        self,
        *,
        model: object,
        model_name: str,
        model_version: str,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.preprocessor = Preprocessor()

    def predict(self, request: PredictInputDTO) -> PredictResultDTO:
        """거래 한 건을 전처리해 예측·기여도를 반환한다."""

        encoded_data = self.preprocessor.predict_preprocess(request)
        prediction = predict(self.model, encoded_data)
        shap_values = shap_decode(prediction["shap_values"])

        return PredictResultDTO(
            predict_result=prediction["predict_result"],
            predict_proba=prediction["predict_proba"],
            shap_values=shap_values,
            model_name=self.model_name,
            model_version=self.model_version,
        )
