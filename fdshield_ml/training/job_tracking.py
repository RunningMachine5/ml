"""Training Job의 데이터 검증 결과를 MLflow Run으로 기록한다.

실제 모델 학습 전에도 Cloud Run Job에서 GCS 입력과 MLflow Tracking Server를
끝까지 연결할 수 있는지 확인하기 위한 스텁이다. 원본 거래 행은 업로드하지 않고
행·컬럼 수와 라벨 분포 같은 요약 정보만 기록한다.
"""

from __future__ import annotations

from dataclasses import asdict

import mlflow

from fdshield_ml.training.data_loader import TrainingDataSummary
from fdshield_ml.training.tracking import configure_tracking, verify_connection


class TrainingTrackingError(RuntimeError):
    """MLflow 설정, 인증, 네트워크 또는 Run 기록에 실패한 경우."""


def log_data_validation_run(
    job_type: str,
    data_uri: str,
    experiment_name: str,
    source_type: str,
    summary: TrainingDataSummary,
) -> str:
    """검증된 학습 데이터의 요약을 하나의 MLflow Run으로 기록한다.

    Cloud Run에서는 환경변수와 Secret Manager로 주입된 MLflow 접속 정보를
    사용한다. 비밀번호나 원본 거래 데이터는 Parameter, Tag, Artifact에 기록하지
    않는다.
    """

    try:
        configure_tracking(None)
        verify_connection()
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{job_type}-data-validation") as run:
            mlflow.log_params(
                {
                    "job_type": job_type,
                    "mode": "data-validation",
                    "data_source": source_type,
                    "data_uri": data_uri,
                    "target_mapping": "m=0,a-l=1",
                }
            )
            mlflow.log_metrics(asdict(summary))
            mlflow.set_tags(
                {
                    "project": "fdshield",
                    "task": "binary_fraud_detection",
                    "pipeline_stage": "stub",
                    "validation_status": "validated",
                    "tracking_client": "cloud_run_training_job",
                }
            )
            mlflow.log_dict(
                {
                    "job_type": job_type,
                    "data_uri": data_uri,
                    "source_type": source_type,
                    "summary": asdict(summary),
                },
                "validation/training-data-summary.json",
            )
            return run.info.run_id
    except Exception as error:
        raise TrainingTrackingError(
            "Failed to record the training data validation in MLflow. "
            "Check the tracking URI, credentials, network, and server status."
        ) from error
