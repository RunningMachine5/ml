"""
[ML 예측 서비스]
컨트롤러 -> 전처리 -> 추론 -> 후처리 -> 컨트롤러 흐름을 관장한다.
"""

from __future__ import annotations

from fdshield_ml.dto.predict_input import PredictInputDTO
from fdshield_ml.dto.predict_result import PredictResultDTO
from fdshield_ml.service.predict.binary_classifier import predict
from fdshield_ml.service.predict.predict_postprocessor import shap_decode
from fdshield_ml.service.preprocessor import Preprocessor


class PredictionServiceError(RuntimeError):
    """모델 파일을 불러오지 못했을 때 발생한다."""


def ml_predict_flow(
    transaction: PredictInputDTO,
    preprocessor: Preprocessor,
    model: object,
) -> dict[str, object]:
    """doo 원본과 같은 순서로 거래 한 건의 예측 흐름을 실행한다."""

    # 이 함수에는 HTTP, MLflow 같은 운영 의존성을 넣지 않는다. 따라서 doo가
    # 로컬에서 모델 로직만 확인할 때도 같은 흐름을 그대로 호출할 수 있다.
    encoded_data = preprocessor.predict_preprocess(transaction)
    prediction = predict(model, encoded_data)
    prediction["shap_values"] = shap_decode(prediction["shap_values"])
    return prediction


class PredictService:
    """doo 예측 결과에 운영용 모델 이름과 버전을 함께 담는다."""

    def __init__(
        self,
        *,
        model: object,
        model_name: str,
        model_version: str,
    ) -> None:
        # XGBoost 파일에는 Registry 이름과 배포 버전이 없으므로 모델 로더가
        # 확인한 값을 모델과 함께 보관한다.
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.preprocessor = Preprocessor()

    def predict(self, request: PredictInputDTO) -> PredictResultDTO:
        """거래 한 건을 전처리해 예측·기여도를 반환한다."""

        prediction = ml_predict_flow(request, self.preprocessor, self.model)

        return PredictResultDTO(
            predict_result=prediction["predict_result"],
            predict_proba=prediction["predict_proba"],
            shap_values=prediction["shap_values"],
            model_name=self.model_name,
            model_version=self.model_version,
        )
