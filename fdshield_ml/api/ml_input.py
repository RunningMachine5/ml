"""
[ML 서버 컨트롤러]
거래 피처를 받아 예측 결과와 피처 기여도를 반환한다.
"""

from fastapi import APIRouter, Request

from fdshield_ml.dto.predict_input import PredictInputDTO
from fdshield_ml.dto.predict_result import PredictResultDTO
from fdshield_ml.service.predict.predict_service import PredictService

router = APIRouter(prefix="/ml", tags=["machine learning"])


@router.post("/predict", response_model=PredictResultDTO)
def ml_input(
    transaction: PredictInputDTO,
    request: Request,
) -> PredictResultDTO:
    """Backend의 raw51 요청을 모델 예측 흐름에 전달한다."""

    service: PredictService = request.app.state.predict_service
    return service.predict(transaction)
