"""공용 MLflow Tracking Server 연결을 설정하고 확인하는 보조 모듈.

이 파일은 모델을 학습하거나 결과를 자동으로 기록하지 않는다. ``.env.tracking``의
서버 주소와 인증 정보를 MLflow 클라이언트에 설정하는 역할만 담당한다. 실제 Run,
파라미터, 지표, 모델 기록 방법은 ``train_xgboost.py`` 예제를 참고한다.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


def configure_tracking(env_file: str | Path | None = ".env.tracking") -> str:
    """환경 파일을 읽고 현재 Python 프로세스의 MLflow 접속 정보를 설정한다.

    Args:
        env_file: MLflow 서버 주소와 계정이 들어 있는 환경 파일 경로. ``None``이면
            이미 설정된 프로세스 환경변수만 사용한다.

    Returns:
        정규화된 MLflow Tracking Server 주소.

    Raises:
        FileNotFoundError: 지정한 환경 파일이 없을 때.
        ValueError: 서버 주소나 Basic Auth 계정 정보가 빠졌을 때.
    """

    path = Path(env_file) if env_file else None
    if path:
        if not path.is_file():
            raise FileNotFoundError(
                f"Tracking environment file not found: {path}. "
                "Copy .env.tracking.example to .env.tracking first."
            )
        # 셸이나 CI가 이미 주입한 값이 있으면 로컬 파일이 덮어쓰지 않도록 한다.
        load_dotenv(path, override=False)

    # rstrip("/")로 주소 끝의 슬래시 유무에 따른 URL 중복을 방지한다.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip().rstrip("/")
    username = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
    password = os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()

    # 학습을 오래 실행한 뒤 인증 오류를 발견하지 않도록 필수 값을 먼저 검사한다.
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is required.")
    if not tracking_uri.startswith(("http://", "https://")):
        raise ValueError("MLFLOW_TRACKING_URI must be an HTTP(S) URL.")
    if not username or not password:
        raise ValueError(
            "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD are required."
        )
    # 이후 호출되는 mlflow.start_run(), MlflowClient() 등이 이 서버를 사용한다.
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def verify_connection() -> int:
    """실험 목록을 한 번 조회하여 네트워크와 계정 권한을 함께 확인한다."""

    # 단순 HTTP 접속뿐 아니라 MLflow API 인증까지 성공해야 목록이 반환된다.
    experiments = MlflowClient().search_experiments(max_results=10)
    return len(experiments)


def main() -> None:
    """``fdshield-mlflow-check`` 명령의 실행 진입점."""

    parser = argparse.ArgumentParser(
        description="Check access to the shared FDShield MLflow server."
    )
    parser.add_argument("--env-file", default=".env.tracking")
    args = parser.parse_args()
    tracking_uri = configure_tracking(args.env_file)
    visible_count = verify_connection()
    print(f"MLflow connection OK: {tracking_uri}")
    print(f"Visible experiments (up to 10): {visible_count}")
