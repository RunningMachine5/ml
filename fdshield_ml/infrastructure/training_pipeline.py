"""ML core 결과를 MLflow Registry와 Backend 운영 흐름에 연결한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fdshield_ml.infrastructure import mlflow as mlflow_integration
from fdshield_ml.service.train.model_training import ModelTrainingConfig
from fdshield_ml.service.train.train_service import (
    TrainingServiceError,
    ml_train_flow,
)


class ProductionTrainingError(RuntimeError):
    """운영 모델 학습·평가·등록 단계가 완료되지 못했을 때 발생한다."""


@dataclass(frozen=True, kw_only=True)
class ProductionTrainingConfig:
    """순수 모델 설정과 MLflow 후보 비교 정책."""

    registered_model_name: str
    model_alias: str = "champion"
    minimum_pr_auc: float = 0.0
    minimum_recall: float = 0.0
    model: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)

    def __post_init__(self) -> None:
        if not self.registered_model_name.strip():
            raise ValueError("registered_model_name is required")
        if not self.model_alias.strip():
            raise ValueError("model_alias is required")
        for name in ("minimum_pr_auc", "minimum_recall"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class ProductionTrainingResult:
    run_id: str
    model_version: int
    metrics: dict[str, float]
    validation_passed: bool
    recommendation: str
    champion_model_version: int | None
    champion_metrics: dict[str, float] | None


def promotion_recommendation(
    candidate: dict[str, float],
    champion: dict[str, float] | None,
    *,
    validation_passed: bool,
) -> str:
    """자동 배포가 아닌 관리자 검토용 상대 비교 결과를 만든다."""

    if not validation_passed:
        return "NOT_RECOMMENDED"
    if champion is None:
        return "REVIEW_REQUIRED"
    if (
        candidate["validation_pr_auc"] > champion["validation_pr_auc"]
        and candidate["validation_recall"] >= champion["validation_recall"]
        and candidate["validation_fpr"] <= champion["validation_fpr"]
    ):
        return "RECOMMENDED"
    if candidate["validation_pr_auc"] > champion["validation_pr_auc"]:
        return "REVIEW_REQUIRED"
    return "NOT_RECOMMENDED"


def run_production_training(
    data_path: str | Path,
    experiment_name: str,
    config: ProductionTrainingConfig,
) -> ProductionTrainingResult:
    """같은 core 학습 결과를 비교하고 MLflow 후보로 등록한다."""

    # 위쪽 service/train은 모델 학습만 담당한다. 이 파일에서만 MLflow 비교와
    # Registry 등록을 붙여 doo의 학습 코드가 운영 환경에 의존하지 않게 한다.
    try:
        mlflow_integration.configure_tracking(None)
        candidate = ml_train_flow(data_path, config.model)
        client = mlflow_integration.MlflowClient()
        champion = mlflow_integration.evaluate_champion(
            client,
            registered_model_name=config.registered_model_name,
            model_alias=config.model_alias,
            features=candidate.validation_features,
            target=candidate.validation_target,
        )
        validation_passed = (
            candidate.metrics["validation_pr_auc"] >= config.minimum_pr_auc
            and candidate.metrics["validation_recall"] >= config.minimum_recall
        )
        recommendation = promotion_recommendation(
            candidate.metrics,
            champion.metrics,
            validation_passed=validation_passed,
        )
        registered = mlflow_integration.register_candidate(
            client,
            candidate,
            experiment_name=experiment_name,
            registered_model_name=config.registered_model_name,
            model_config=config.model,
            minimum_pr_auc=config.minimum_pr_auc,
            minimum_recall=config.minimum_recall,
            validation_passed=validation_passed,
            recommendation=recommendation,
            champion=champion,
        )
    except TrainingServiceError as exc:
        raise ProductionTrainingError(str(exc)) from exc
    except mlflow_integration.MlflowIntegrationError as exc:
        raise ProductionTrainingError(str(exc)) from exc

    return ProductionTrainingResult(
        run_id=registered.run_id,
        model_version=registered.model_version,
        metrics=candidate.metrics,
        validation_passed=validation_passed,
        recommendation=recommendation,
        champion_model_version=champion.model_version,
        champion_metrics=champion.metrics,
    )
