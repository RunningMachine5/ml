"""로컬 Docker와 Cloud Run에서 공통으로 사용하는 서빙 실행 진입점."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Cloud Run이 주입하는 PORT로 FastAPI 서버를 실행한다."""

    port = int(os.getenv("PORT", "8080"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535.")

    uvicorn.run(
        "fdshield_ml.serving.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
