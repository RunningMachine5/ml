"""로컬 Docker와 Cloud Run에서 공통으로 사용하는 FastAPI 애플리케이션."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from fdshield_ml.serving.api.ml_input import router as prediction_router
from fdshield_ml.serving.dto.predict_result import HealthResponse
from fdshield_ml.serving.integrations import predict_service_from_environment
from fdshield_ml.serving.service.predict.predict_service import PredictService


def create_app(predict_service: PredictService | None = None) -> FastAPI:
    """테스트와 운영 모델 서비스를 교체할 수 있는 앱을 생성한다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 실제 모드에서는 시작 시 고정 모델 버전을 모두 로딩한다. 실패하면
        # Cloud Run 신규 Revision이 ready 상태가 되지 않는다. 모듈 import와
        # OpenAPI 생성 자체는 모델 파일 없이도 가능하게 시작 단계로 분리한다.
        if predict_service is None:
            app.state.predict_service = predict_service_from_environment()
        yield

    app = FastAPI(
        title="FDShield ML Serving",
        version="0.2.0",
        lifespan=lifespan,
    )
    if predict_service is not None:
        app.state.predict_service = predict_service
    app.include_router(prediction_router)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=HealthResponse)
    def ready(request: Request) -> HealthResponse:
        service: PredictService = request.app.state.predict_service
        if not service.ready:
            raise HTTPException(status_code=503, detail="model is not ready")
        return HealthResponse(status="ready")

    return app


app = create_app()
