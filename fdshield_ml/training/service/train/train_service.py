"""raw64 데이터 전처리부터 MLflow 후보 등록까지 학습 흐름을 관리한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fdshield_ml.common.preprocess_config import MODEL_FEATURE_COLUMNS
from fdshield_ml.common.preprocessor import FeaturePreprocessingError, Preprocessor
from fdshield_ml.training.dataset import (
    TrainingDatasetError,
    normalize_training_frame,
    validate_binary_target,
    validate_transaction_ids,
)
from fdshield_ml.training.integrations import mlflow as mlflow_integration
from fdshield_ml.training.service.train.model_training import (
    ModelTrainingConfig,
    ModelTrainingError,
    train_model,
)


class ProductionTrainingError(RuntimeError):
    """운영 모델 학습·평가·등록 단계가 완료되지 못했을 때 발생한다."""


@dataclass(frozen=True, kw_only=True)
class ProductionTrainingConfig(ModelTrainingConfig):
    """모델 학습 설정과 MLflow 후보 비교 정책."""

    registered_model_name: str
    model_alias: str = "champion"
    minimum_pr_auc: float = 0.0
    minimum_recall: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.registered_model_name.strip():
            raise ValueError("registered_model_name is required")
        if not self.model_alias.strip():
            raise ValueError("model_alias is required")
        for name in ("minimum_pr_auc", "minimum_recall"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def model_training_config(self) -> ModelTrainingConfig:
        return ModelTrainingConfig(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            scale_pos_weight=self.scale_pos_weight,
            tree_method=self.tree_method,
            early_stopping_rounds=self.early_stopping_rounds,
            n_jobs=self.n_jobs,
        )


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


def ml_train_flow(
    data_path: str | Path,
    experiment_name: str,
    config: ProductionTrainingConfig,
) -> ProductionTrainingResult:
    """train1 raw64를 전처리·학습하고 MLflow 후보로 등록한다."""

    mlflow_integration.configure_tracking(None)
    mlflow_integration.verify_connection()

    try:
        source = pd.read_csv(Path(data_path), low_memory=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ProductionTrainingError(
            f"Failed to read training CSV: {data_path}"
        ) from exc
    if len(source) < 10:
        raise ProductionTrainingError("Training data must contain at least 10 rows.")

    try:
        normalized = normalize_training_frame(source)
        target = validate_binary_target(normalized, context="training")
        validate_transaction_ids(normalized, context="training")
        train_features = Preprocessor().train_preprocess(normalized)
    except FeaturePreprocessingError as exc:
        raise ProductionTrainingError(f"Invalid train1 raw64 features: {exc}") from exc
    except TrainingDatasetError as exc:
        raise ProductionTrainingError(str(exc)) from exc

    if train_features.columns.tolist() != list(MODEL_FEATURE_COLUMNS):
        raise ProductionTrainingError(
            "Preprocessed training schema is not the model80 contract."
        )

    model_config = config.model_training_config()
    try:
        candidate = train_model(train_features, target, model_config)
        client = mlflow_integration.create_registry_client()
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
            model_config=model_config,
            minimum_pr_auc=config.minimum_pr_auc,
            minimum_recall=config.minimum_recall,
            validation_passed=validation_passed,
            recommendation=recommendation,
            champion=champion,
        )
    except ModelTrainingError as exc:
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


__all__ = [
    "ProductionTrainingConfig",
    "ProductionTrainingError",
    "ProductionTrainingResult",
    "ml_train_flow",
    "promotion_recommendation",
]
