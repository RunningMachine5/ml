"""FDShield 모델 서빙 FastAPI 애플리케이션."""

from fastapi import FastAPI, HTTPException, Request
from fdshield_ml.common.preprocessing import FeaturePreprocessingError
from fdshield_ml.serving.model_predictor import ModelServingError
from fdshield_ml.serving.predictor import Predictor, predictor_from_environment
from fdshield_ml.serving.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """테스트 예측기와 실제 예측기를 교체할 수 있는 앱을 생성한다."""

    app = FastAPI(title="FDShield ML Serving", version="0.1.0")
    # 실제 모드에서는 이 시점에 고정 MLflow 모델 버전을 모두 로딩한다. 실패하면
    # 프로세스가 포트를 열지 못하므로 Cloud Run 신규 Revision이 준비되지 않는다.
    app.state.predictor = predictor or predictor_from_environment()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=HealthResponse)
    def ready(request: Request) -> HealthResponse:
        serving_predictor: Predictor = request.app.state.predictor
        if not serving_predictor.ready:
            raise HTTPException(status_code=503, detail="model is not ready")
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
        except ModelServingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
