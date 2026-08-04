"""FDShield 모델 서빙 FastAPI 애플리케이션."""

from fastapi import FastAPI, Request

from fdshield_ml.serving.predictor import Predictor, StubPredictor
from fdshield_ml.serving.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """테스트 예측기와 실제 예측기를 교체할 수 있는 앱을 생성한다."""

    app = FastAPI(title="FDShield ML Serving", version="0.1.0")
    app.state.predictor = predictor or StubPredictor()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=HealthResponse)
    def ready() -> HealthResponse:
        return HealthResponse(status="ready")

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        serving_predictor: Predictor = request.app.state.predictor
        return serving_predictor.predict(payload)

    return app


app = create_app()
