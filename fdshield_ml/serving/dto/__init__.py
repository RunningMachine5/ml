"""ML 추론 요청·응답 DTO."""

from fdshield_ml.serving.dto.predict_input import PredictInputDTO
from fdshield_ml.serving.dto.predict_result import HealthResponse, PredictResultDTO

__all__ = ["HealthResponse", "PredictInputDTO", "PredictResultDTO"]
