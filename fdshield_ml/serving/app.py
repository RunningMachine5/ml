"""FDShield 모델 서빙 FastAPI 애플리케이션."""

from fastapi import FastAPI, HTTPException, Request
from fdshield_ml.common.preprocessing import FeaturePreprocessingError
from fdshield_ml.serving.predictor import Predictor, StubPredictor
from fdshield_ml.serving.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """테스트 예측기와 실제 예측기를 교체할 수 있는 앱을 생성한다."""

    app = FastAPI(title="FDShield ML Serving", version="0.1.0")
    # 현재 스켈레톤은 항상 Stub을 사용하고, 실제 모델 확정 후 구현만 교체한다.
    app.state.predictor = predictor or StubPredictor.from_environment()

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
        try:
            return serving_predictor.predict(payload)
        except FeaturePreprocessingError as exc:
            # 형식은 맞지만 전처리할 수 없는 값도 서버 오류가 아니라 요청 오류다.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
