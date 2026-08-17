"""거래를 받아 예측 결과와 Feature 기여도를 반환하는 ML API."""

from fastapi import APIRouter, HTTPException, Request

from fdshield_ml.dto.predict_input import PredictInputDTO
from fdshield_ml.dto.predict_result import PredictResultDTO
from fdshield_ml.service.predict.predict_service import (
    PredictionServiceError,
    PredictService,
    ml_predict_flow,
)

router = APIRouter(prefix="/ml", tags=["machine learning"])


@router.post("/predict", response_model=PredictResultDTO)
def ml_input(
    transaction: PredictInputDTO,
    request: Request,
) -> PredictResultDTO:
    """ML 담당자의 flat raw52 요청을 받아 model79 예측 결과를 반환한다."""

    service: PredictService = request.app.state.predict_service
    try:
        return ml_predict_flow(transaction, service)
    except PredictionServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


__all__ = ["router"]
