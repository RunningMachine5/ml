"""운영용 이진 사기 모델을 학습하고 MLflow Registry에 등록한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from fdshield_ml.common.decision_threshold import (
    DECISION_THRESHOLD_TAG,
    resolve_model_decision_threshold,
    store_model_decision_threshold,
    validate_decision_threshold,
)
from fdshield_ml.common.feature_contract import (
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
)
from fdshield_ml.common.preprocessing import (
    FeaturePreprocessingError,
    preprocess_frame,
)
from fdshield_ml.common.xgboost_prediction import prediction_iteration_range
from fdshield_ml.training.dataset import (
    DEFAULT_MIN_FRAUD_ROWS_PER_SPLIT,
    DEFAULT_SPLIT_DATETIME,
    LABEL_COLUMN,
    TRANSACTION_DATETIME_COLUMN,
    TrainingDatasetError,
    aligned_transaction_datetimes,
    detect_training_dataset_kind,
    time_split_indices,
    transaction_datetimes,
    validate_binary_target,
    validate_preprocessed_features,
)
from fdshield_ml.training.pipeline import best_f1_threshold
from fdshield_ml.training.tracking import configure_tracking, verify_connection


class ProductionTrainingError(RuntimeError):
    """운영 모델 학습·평가·등록 단계가 완료되지 못했을 때 발생한다."""


@dataclass(frozen=True)
class ProductionTrainingConfig:
    registered_model_name: str
    model_alias: str = "champion"
    minimum_pr_auc: float = 0.0
    minimum_recall: float = 0.0
    champion_model_version: int | None = None
    random_state: int = 42
    split_datetime: datetime | str = DEFAULT_SPLIT_DATETIME
    minimum_fraud_rows_per_split: int = DEFAULT_MIN_FRAUD_ROWS_PER_SPLIT
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    early_stopping_rounds: int = 50
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if not self.registered_model_name.strip():
            raise ValueError("registered_model_name is required")
        if not self.model_alias.strip():
            raise ValueError("model_alias is required")
        for name in ("minimum_pr_auc", "minimum_recall"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.champion_model_version is not None and self.champion_model_version < 1:
            raise ValueError("champion_model_version must be positive")
        if self.minimum_fraud_rows_per_split < 1:
            raise ValueError("minimum_fraud_rows_per_split must be positive")
        try:
            split_datetime = pd.Timestamp(self.split_datetime)
        except (TypeError, ValueError) as exc:
            raise ValueError("split_datetime must be a valid datetime") from exc
        if pd.isna(split_datetime):
            raise ValueError("split_datetime must be a valid datetime")
        if self.n_estimators < 1 or self.max_depth < 1:
            raise ValueError("model size and job counts must be positive")
        if self.min_child_weight < 0 or self.reg_lambda < 0:
            raise ValueError("XGBoost regularization values must not be negative")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        if self.tree_method != "hist":
            raise ValueError("tree_method must be hist for production training")


@dataclass(frozen=True)
class ProductionTrainingResult:
    run_id: str
    model_version: int
    metrics: dict[str, float]
    validation_passed: bool
    recommendation: str
    champion_model_version: int | None
    champion_metrics: dict[str, float] | None


def _read_companion_transactions(data_path: str | Path) -> pd.DataFrame:
    """시간 분할에 필요한 원본 거래 시각과 라벨만 읽는다."""

    path = Path(data_path)
    try:
        columns = pd.read_csv(path, nrows=0).columns
        required = {TRANSACTION_DATETIME_COLUMN, LABEL_COLUMN}
        missing = sorted(required - set(columns))
        if missing:
            raise ProductionTrainingError(
                f"Companion transactions data is missing columns: {missing}"
            )
        return pd.read_csv(
            path,
            usecols=[TRANSACTION_DATETIME_COLUMN, LABEL_COLUMN],
            low_memory=False,
        )
    except ProductionTrainingError:
        raise
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise ProductionTrainingError(
            f"Failed to read companion transactions CSV: {data_path}"
        ) from exc


def _evaluation_metrics(
    target: pd.Series,
    probability: pd.Series,
    threshold: float,
) -> dict[str, float]:
    threshold = validate_decision_threshold(
        threshold,
        source="evaluation decision threshold",
    )
    predicted = probability.ge(threshold).astype("int8")
    tn, fp, _, _ = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "validation_pr_auc": float(average_precision_score(target, probability)),
        "validation_roc_auc": float(roc_auc_score(target, probability)),
        "validation_recall": float(recall_score(target, predicted, zero_division=0)),
        "validation_precision": float(
            precision_score(target, predicted, zero_division=0)
        ),
        "validation_f1": float(f1_score(target, predicted, zero_division=0)),
        "validation_fpr": float(false_positive_rate),
        "decision_threshold": threshold,
    }


def _champion_evaluation(
    client: MlflowClient,
    config: ProductionTrainingConfig,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[int | None, dict[str, float] | None]:
    """현재 champion을 후보와 동일한 검증 행에서 다시 평가한다."""

    if config.champion_model_version is not None:
        champion_version = config.champion_model_version
    else:
        try:
            version = client.get_model_version_by_alias(
                config.registered_model_name,
                config.model_alias,
            )
            champion_version = int(version.version)
        except Exception as exc:
            if getattr(exc, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
                return None, None
            raise ProductionTrainingError("Failed to resolve the current champion model.") from exc

    try:
        champion = mlflow.sklearn.load_model(
            f"models:/{config.registered_model_name}/{champion_version}"
        )
        version = client.get_model_version(
            config.registered_model_name,
            str(champion_version),
        )
        threshold = resolve_model_decision_threshold(
            champion,
            model_version_tags=getattr(version, "tags", None),
        )
        probability = pd.Series(
            champion.predict_proba(features)[:, 1],
            index=target.index,
        )
    except Exception as exc:
        raise ProductionTrainingError(
            "Failed to evaluate the current champion model."
        ) from exc
    return champion_version, _evaluation_metrics(target, probability, threshold)


def _promotion_recommendation(
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


def train_and_register_model(
    data_path: str | Path,
    experiment_name: str,
    config: ProductionTrainingConfig,
    transactions_path: str | Path | None = None,
) -> ProductionTrainingResult:
    """고정 전처리로 XGBoost를 학습하고 Registry 버전과 승인 상태를 기록한다."""

    configure_tracking(None)
    verify_connection()

    try:
        source = pd.read_csv(Path(data_path), low_memory=False).reset_index(drop=True)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ProductionTrainingError(f"Failed to read training CSV: {data_path}") from exc

    if len(source) < 10:
        raise ProductionTrainingError("Training data must contain at least 10 rows.")

    try:
        dataset_kind = detect_training_dataset_kind(source.columns)
        target = validate_binary_target(source, context="training")
        if dataset_kind == "raw":
            numeric_features = preprocess_frame(source.loc[:, MODEL_INPUT_COLUMNS])
            datetimes = transaction_datetimes(source, context="training")
        else:
            if transactions_path is None:
                raise TrainingDatasetError(
                    "preprocessed training data requires companion transactions data"
                )
            numeric_features = validate_preprocessed_features(source)
            transactions = _read_companion_transactions(transactions_path)
            datetimes = aligned_transaction_datetimes(source, transactions)
        split = time_split_indices(
            datetimes,
            target,
            split_datetime=config.split_datetime,
            minimum_fraud_rows=config.minimum_fraud_rows_per_split,
        )
        train_index = split.train_index
        validation_index = split.validation_index
        split_metadata: dict[str, object] = {
            "split_strategy": "time",
            "split_datetime": split.boundary.isoformat(sep=" "),
            "train_datetime_min": split.train_datetime_min.isoformat(sep=" "),
            "train_datetime_max": split.train_datetime_max.isoformat(sep=" "),
            "validation_datetime_min": split.validation_datetime_min.isoformat(
                sep=" "
            ),
            "validation_datetime_max": split.validation_datetime_max.isoformat(
                sep=" "
            ),
            "train_fraud_rows": split.train_fraud_count,
            "validation_fraud_rows": split.validation_fraud_count,
        }
    except FeaturePreprocessingError as exc:
        raise ProductionTrainingError(f"Invalid raw training features: {exc}") from exc
    except TrainingDatasetError as exc:
        raise ProductionTrainingError(str(exc)) from exc

    if numeric_features.columns.tolist() != list(MODEL_FEATURE_COLUMNS):
        raise ProductionTrainingError("Preprocessed training schema is not the 91-column contract.")

    X_train = numeric_features.loc[train_index]
    X_validation = numeric_features.loc[validation_index]
    y_train = target.loc[train_index]
    y_validation = target.loc[validation_index]
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ProductionTrainingError(
            "Training split must contain both normal and fraud rows."
        )

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        tree_method=config.tree_method,
        early_stopping_rounds=config.early_stopping_rounds,
        scale_pos_weight=negative_count / positive_count,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    classifier.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    candidate_iteration_range = prediction_iteration_range(
        classifier,
        classifier.get_booster(),
    )
    probability = pd.Series(
        classifier.predict_proba(
            X_validation,
            iteration_range=candidate_iteration_range,
        )[:, 1],
        index=y_validation.index,
    )
    decision_threshold = best_f1_threshold(
        y_validation,
        probability.to_numpy(),
    )
    store_model_decision_threshold(classifier, decision_threshold)
    metrics = _evaluation_metrics(y_validation, probability, decision_threshold)
    validation_passed = (
        metrics["validation_pr_auc"] >= config.minimum_pr_auc
        and metrics["validation_recall"] >= config.minimum_recall
    )
    client = MlflowClient()
    champion_model_version, champion_metrics = _champion_evaluation(
        client,
        config,
        X_validation,
        y_validation,
    )
    recommendation = _promotion_recommendation(
        metrics,
        champion_metrics,
        validation_passed=validation_passed,
    )
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="cloud-run-production-training") as run:
        mlflow.log_params(
            {
                "model_type": "xgboost",
                "training_data_contract": dataset_kind,
                "feature_contract": (
                    "fdshield-raw-54-to-model-91"
                    if dataset_kind == "raw"
                    else "fdshield-preprocessed-model-91"
                ),
                "train_rows": len(X_train),
                "validation_rows": len(X_validation),
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "learning_rate": config.learning_rate,
                "min_child_weight": config.min_child_weight,
                "subsample": config.subsample,
                "colsample_bytree": config.colsample_bytree,
                "reg_lambda": config.reg_lambda,
                "tree_method": config.tree_method,
                "early_stopping_rounds": config.early_stopping_rounds,
                "random_state": config.random_state,
                "minimum_pr_auc": config.minimum_pr_auc,
                "minimum_recall": config.minimum_recall,
                "decision_threshold": decision_threshold,
                **split_metadata,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "project": "fdshield",
                "task": "binary_fraud_detection",
                "pipeline_stage": "production_training",
                "validation_status": "passed" if validation_passed else "failed",
                "promotion_recommendation": recommendation,
                "champion_model_version": str(champion_model_version or ""),
            }
        )
        mlflow.log_dict(
            {
                "candidate": metrics,
                "champion_model_version": champion_model_version,
                "champion": champion_metrics,
                "recommendation": recommendation,
            },
            "metadata/model-comparison.json",
        )
        mlflow.log_dict(
            {"model_feature_columns": list(MODEL_FEATURE_COLUMNS)},
            "metadata/model-feature-schema.json",
        )
        input_example = X_validation.head(5)
        output_example = classifier.predict_proba(
            input_example,
            iteration_range=candidate_iteration_range,
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=classifier,
            name="model",
            signature=infer_signature(input_example, output_example),
            input_example=input_example,
            serialization_format="cloudpickle",
            pyfunc_predict_fn="predict_proba",
            registered_model_name=config.registered_model_name,
        )
        run_id = run.info.run_id

    model_version = model_info.registered_model_version
    if model_version is None:
        raise ProductionTrainingError("MLflow did not return a registered model version.")

    client.set_model_version_tag(
        config.registered_model_name,
        str(model_version),
        "validation_status",
        "passed" if validation_passed else "failed",
    )
    client.set_model_version_tag(
        config.registered_model_name,
        str(model_version),
        "promotion_recommendation",
        recommendation,
    )
    client.set_model_version_tag(
        config.registered_model_name,
        str(model_version),
        DECISION_THRESHOLD_TAG,
        repr(decision_threshold),
    )
    return ProductionTrainingResult(
        run_id=run_id,
        model_version=int(model_version),
        metrics=metrics,
        validation_passed=validation_passed,
        recommendation=recommendation,
        champion_model_version=champion_model_version,
        champion_metrics=champion_metrics,
    )
