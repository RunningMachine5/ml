"""MLflow 연결, 운영 모델 비교, 후보 Registry 등록을 담당한다."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from fdshield_ml.config.preprocess_config import MODEL_FEATURE_COLUMNS
from fdshield_ml.service.train.dataset import TRAINING_DATA_CONTRACT
from fdshield_ml.service.train.model_training import (
    VALIDATION_FRACTION,
    ModelTrainingConfig,
    ModelTrainingResult,
    evaluation_metrics,
)


class MlflowIntegrationError(RuntimeError):
    """운영 모델 조회 또는 후보 Registry 등록이 실패했을 때 발생한다."""


@dataclass(frozen=True)
class ChampionEvaluation:
    """현재 champion 버전과 동일 검증 데이터에서 계산한 지표."""

    model_version: int | None
    metrics: dict[str, float] | None

    @property
    def comparison_status(self) -> str:
        if self.model_version is None:
            return "not_available"
        if self.metrics is None:
            return "skipped_contract_mismatch"
        return "compared"


@dataclass(frozen=True)
class RegisteredCandidate:
    """MLflow가 반환한 후보 Run과 Registry 버전."""

    run_id: str
    model_version: int


def configure_tracking(env_file: str | Path | None = ".env.tracking") -> str:
    """환경 파일 또는 프로세스 환경변수로 MLflow 접속 정보를 설정한다."""

    path = Path(env_file) if env_file else None
    if path:
        if not path.is_file():
            raise FileNotFoundError(
                f"Tracking environment file not found: {path}. "
                "Copy .env.tracking.example to .env.tracking first."
            )
        load_dotenv(path, override=False)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip().rstrip("/")
    username = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
    password = os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is required.")
    if not tracking_uri.startswith(("http://", "https://")):
        raise ValueError("MLFLOW_TRACKING_URI must be an HTTP(S) URL.")
    if not username or not password:
        raise ValueError(
            "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD are required."
        )
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def verify_connection() -> int:
    """실험 목록을 조회하여 MLflow 네트워크와 계정 권한을 확인한다."""

    return len(MlflowClient().search_experiments(max_results=10))


def _model_feature_names(model: object) -> tuple[str, ...] | None:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return tuple(str(name) for name in names)
    try:
        booster_names = model.get_booster().feature_names  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return None
    if booster_names is None:
        return None
    return tuple(str(name) for name in booster_names)


def champion_contract_matches(model: object) -> bool:
    """현재 champion이 같은 model79 입력 계약인지 확인한다."""

    names = _model_feature_names(model)
    if names is not None:
        return names == tuple(MODEL_FEATURE_COLUMNS)
    feature_count = getattr(model, "n_features_in_", None)
    return feature_count is None or int(feature_count) == len(MODEL_FEATURE_COLUMNS)


def evaluate_champion(
    client: MlflowClient,
    *,
    registered_model_name: str,
    model_alias: str,
    features: pd.DataFrame,
    target: pd.Series,
) -> ChampionEvaluation:
    """같은 검증 행에서 model79 champion을 평가한다."""

    try:
        version = client.get_model_version_by_alias(
            registered_model_name,
            model_alias,
        )
        champion_version = int(version.version)
    except Exception as exc:
        if getattr(exc, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
            return ChampionEvaluation(None, None)
        raise MlflowIntegrationError(
            "Failed to resolve the current champion model."
        ) from exc

    try:
        champion = mlflow.sklearn.load_model(
            f"models:/{registered_model_name}/{champion_version}"
        )
        if not champion_contract_matches(champion):
            return ChampionEvaluation(champion_version, None)
        probability = pd.Series(
            champion.predict_proba(features)[:, 1],  # type: ignore[attr-defined]
            index=target.index,
        )
    except Exception as exc:
        raise MlflowIntegrationError(
            "Failed to evaluate the current champion model."
        ) from exc
    return ChampionEvaluation(
        champion_version,
        evaluation_metrics(target, probability),
    )


def register_candidate(
    client: MlflowClient,
    candidate: ModelTrainingResult,
    *,
    experiment_name: str,
    registered_model_name: str,
    model_config: ModelTrainingConfig,
    minimum_pr_auc: float,
    minimum_recall: float,
    validation_passed: bool,
    recommendation: str,
    champion: ChampionEvaluation,
) -> RegisteredCandidate:
    """성능·비교 자료를 기록하고 후보 모델을 Registry에 등록한다."""

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="cloud-run-production-training") as run:
        mlflow.log_params(
            {
                "model_type": "xgboost",
                "training_data_contract": "train1-raw53",
                "feature_contract": TRAINING_DATA_CONTRACT,
                "split_strategy": "random_stratified_80_20",
                "validation_fraction": VALIDATION_FRACTION,
                "train_rows": candidate.train_rows,
                "validation_rows": candidate.validation_rows,
                "n_estimators": model_config.n_estimators,
                "max_depth": model_config.max_depth,
                "learning_rate": model_config.learning_rate,
                "min_child_weight": model_config.min_child_weight,
                "gamma": model_config.gamma,
                "subsample": model_config.subsample,
                "colsample_bytree": model_config.colsample_bytree,
                "reg_lambda": model_config.reg_lambda,
                "reg_alpha": model_config.reg_alpha,
                "scale_pos_weight": model_config.scale_pos_weight,
                "tree_method": model_config.tree_method,
                "eval_metric": "logloss",
                "early_stopping_rounds": model_config.early_stopping_rounds,
                "random_state": model_config.random_state,
                "minimum_pr_auc": minimum_pr_auc,
                "minimum_recall": minimum_recall,
                "decision_threshold": candidate.decision_threshold,
            }
        )
        mlflow.log_metrics(candidate.metrics)
        mlflow.set_tags(
            {
                "project": "fdshield",
                "task": "binary_fraud_detection",
                "pipeline_stage": "production_training",
                "feature_contract": TRAINING_DATA_CONTRACT,
                "validation_status": "passed" if validation_passed else "failed",
                "promotion_recommendation": recommendation,
                "champion_model_version": str(champion.model_version or ""),
                "champion_comparison_status": champion.comparison_status,
            }
        )
        mlflow.log_dict(
            {
                "candidate": candidate.metrics,
                "champion_model_version": champion.model_version,
                "champion": champion.metrics,
                "champion_comparison_status": champion.comparison_status,
                "recommendation": recommendation,
            },
            "metadata/model-comparison.json",
        )
        mlflow.log_dict(
            {
                "feature_contract": TRAINING_DATA_CONTRACT,
                "feature_count": len(MODEL_FEATURE_COLUMNS),
                "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
            },
            "metadata/model-feature-schema.json",
        )
        input_example = candidate.validation_features.head(5)
        output_example = candidate.model.predict_proba(
            input_example,
            iteration_range=candidate.iteration_range,
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=candidate.model,
            name="model",
            signature=infer_signature(input_example, output_example),
            input_example=input_example,
            serialization_format="cloudpickle",
            pyfunc_predict_fn="predict_proba",
            registered_model_name=registered_model_name,
        )
        run_id = run.info.run_id

    model_version = model_info.registered_model_version
    if model_version is None:
        raise MlflowIntegrationError(
            "MLflow did not return a registered model version."
        )
    version_tags = {
        "validation_status": "passed" if validation_passed else "failed",
        "promotion_recommendation": recommendation,
        "decision_threshold": repr(candidate.decision_threshold),
        "feature_contract": TRAINING_DATA_CONTRACT,
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(
            registered_model_name,
            str(model_version),
            key,
            value,
        )
    return RegisteredCandidate(run_id=run_id, model_version=int(model_version))


def main() -> None:
    """`.env.tracking`을 이용한 공용 MLflow 연결 확인 CLI."""

    parser = argparse.ArgumentParser(
        description="Check access to the shared FDShield MLflow server."
    )
    parser.add_argument("--env-file", default=".env.tracking")
    args = parser.parse_args()
    tracking_uri = configure_tracking(args.env_file)
    visible_count = verify_connection()
    print(f"MLflow connection OK: {tracking_uri}")
    print(f"Visible experiments (up to 10): {visible_count}")


if __name__ == "__main__":
    main()
