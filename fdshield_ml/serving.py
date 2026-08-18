"""로컬 Docker와 Cloud Run에서 공통으로 사용하는 FastAPI 애플리케이션."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from fdshield_ml.api.ml_input import router as prediction_router
from fdshield_ml.dto.predict_result import HealthResponse
from fdshield_ml.infrastructure.model_loader import predict_service_from_environment
from fdshield_ml.service.predict.predict_service import PredictService


def create_app(predict_service: PredictService | None = None) -> FastAPI:
    """테스트와 운영 모델 서비스를 교체할 수 있는 앱을 생성한다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 요청마다 모델을 읽으면 느리므로 서버 시작 시 승인된 한 버전만 로딩한다.
        # 로딩 실패 시 시작도 실패하게 해 잘못된 Revision이 트래픽을 받지 않게 한다.
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
    def ready() -> HealthResponse:
        return HealthResponse(status="ready")

    return app


app = create_app()


def main() -> None:
    """Cloud Run이 주입하는 PORT로 FastAPI 서버를 실행한다."""

    port = int(os.getenv("PORT", "8080"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535.")

    uvicorn.run(
        "fdshield_ml.serving:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
