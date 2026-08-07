"""운영용 이진 사기 모델을 학습하고 MLflow Registry에 등록한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBClassifier

from fdshield_ml.common.feature_contract import (
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
)
from fdshield_ml.common.preprocessing import preprocess_frame
from fdshield_ml.training.tracking import configure_tracking, verify_connection


LABEL_COLUMN = "Is_Fraud"
GROUP_COLUMN = "Account_account_number"


class ProductionTrainingError(RuntimeError):
    """운영 모델 학습·평가·등록 단계가 완료되지 못했을 때 발생한다."""


@dataclass(frozen=True)
class ProductionTrainingConfig:
    registered_model_name: str
    model_alias: str = "champion"
    auto_promote: bool = False
    minimum_pr_auc: float = 0.0
    minimum_recall: float = 0.0
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    n_jobs: int = 2

    def __post_init__(self) -> None:
        if not self.registered_model_name.strip():
            raise ValueError("registered_model_name is required")
        if not self.model_alias.strip():
            raise ValueError("model_alias is required")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        for name in ("minimum_pr_auc", "minimum_recall"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.n_estimators < 1 or self.max_depth < 1 or self.n_jobs < 1:
            raise ValueError("model size and job counts must be positive")


@dataclass(frozen=True)
class ProductionTrainingResult:
    run_id: str
    model_version: int
    metrics: dict[str, float]
    validation_passed: bool
    promoted: bool


def _binary_target(source: pd.DataFrame) -> pd.Series:
    if LABEL_COLUMN not in source:
        raise ProductionTrainingError(f"Training data is missing {LABEL_COLUMN}.")
    try:
        target = pd.to_numeric(source[LABEL_COLUMN], errors="raise").astype("int8")
    except (TypeError, ValueError) as exc:
        raise ProductionTrainingError(f"{LABEL_COLUMN} must contain 0 or 1.") from exc
    invalid = sorted(set(target.unique()) - {0, 1})
    if invalid or target.nunique() != 2:
        raise ProductionTrainingError(
            f"{LABEL_COLUMN} must contain both 0 and 1; invalid={invalid}"
        )
    return target.rename("is_fraud")


def _split_indices(
    source: pd.DataFrame,
    target: pd.Series,
    config: ProductionTrainingConfig,
) -> tuple[pd.Index, pd.Index]:
    if GROUP_COLUMN in source:
        groups = source[GROUP_COLUMN].astype("string")
        if groups.isna().any():
            groups = groups.fillna(
                pd.Series(
                    [f"__missing_group_{index}" for index in source.index],
                    index=source.index,
                    dtype="string",
                )
            )
        splitter = GroupShuffleSplit(
            n_splits=10,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        for train_positions, validation_positions in splitter.split(
            source, target, groups
        ):
            if (
                target.iloc[train_positions].nunique() == 2
                and target.iloc[validation_positions].nunique() == 2
            ):
                return source.index[train_positions], source.index[validation_positions]
        raise ProductionTrainingError(
            "Group split could not place both labels in train and validation data."
        )

    train_index, validation_index = train_test_split(
        source.index,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )
    return pd.Index(train_index), pd.Index(validation_index)


def _evaluation_metrics(
    target: pd.Series,
    probability: pd.Series,
) -> dict[str, float]:
    predicted = probability.ge(0.5).astype("int8")
    return {
        "validation_pr_auc": float(average_precision_score(target, probability)),
        "validation_roc_auc": float(roc_auc_score(target, probability)),
        "validation_recall": float(recall_score(target, predicted, zero_division=0)),
        "validation_precision": float(
            precision_score(target, predicted, zero_division=0)
        ),
        "validation_f1": float(f1_score(target, predicted, zero_division=0)),
    }


def train_and_register_model(
    data_path: str | Path,
    experiment_name: str,
    config: ProductionTrainingConfig,
) -> ProductionTrainingResult:
    """고정 전처리로 XGBoost를 학습하고 Registry 버전과 승인 상태를 기록한다."""

    configure_tracking(None)
    verify_connection()

    try:
        source = pd.read_csv(Path(data_path), low_memory=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ProductionTrainingError(f"Failed to read training CSV: {data_path}") from exc

    missing = sorted(set(MODEL_INPUT_COLUMNS) - set(source.columns))
    if missing:
        raise ProductionTrainingError(f"Training data is missing raw features: {missing}")
    if len(source) < 10:
        raise ProductionTrainingError("Training data must contain at least 10 rows.")

    target = _binary_target(source)
    train_index, validation_index = _split_indices(source, target, config)
    raw_features = source.loc[:, MODEL_INPUT_COLUMNS]
    numeric_features = preprocess_frame(raw_features)
    if numeric_features.columns.tolist() != list(MODEL_FEATURE_COLUMNS):
        raise ProductionTrainingError("Preprocessed training schema is not the 91-column contract.")

    X_train = numeric_features.loc[train_index]
    X_validation = numeric_features.loc[validation_index]
    y_train = target.loc[train_index]
    y_validation = target.loc[validation_index]
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    if positive_count == 0:
        raise ProductionTrainingError("Training split contains no fraud rows.")

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        scale_pos_weight=negative_count / positive_count,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    classifier.fit(X_train, y_train)
    probability = pd.Series(
        classifier.predict_proba(X_validation)[:, 1],
        index=y_validation.index,
    )
    metrics = _evaluation_metrics(y_validation, probability)
    validation_passed = (
        metrics["validation_pr_auc"] >= config.minimum_pr_auc
        and metrics["validation_recall"] >= config.minimum_recall
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="cloud-run-production-training") as run:
        mlflow.log_params(
            {
                "model_type": "xgboost",
                "feature_contract": "fdshield-raw-54-to-model-91",
                "train_rows": len(X_train),
                "validation_rows": len(X_validation),
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "learning_rate": config.learning_rate,
                "subsample": config.subsample,
                "colsample_bytree": config.colsample_bytree,
                "random_state": config.random_state,
                "minimum_pr_auc": config.minimum_pr_auc,
                "minimum_recall": config.minimum_recall,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "project": "fdshield",
                "task": "binary_fraud_detection",
                "pipeline_stage": "production_training",
                "validation_status": "passed" if validation_passed else "failed",
            }
        )
        mlflow.log_dict(
            {"model_feature_columns": list(MODEL_FEATURE_COLUMNS)},
            "metadata/model-feature-schema.json",
        )
        input_example = X_validation.head(5)
        output_example = classifier.predict_proba(input_example)
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

    client = MlflowClient()
    client.set_model_version_tag(
        config.registered_model_name,
        str(model_version),
        "validation_status",
        "passed" if validation_passed else "failed",
    )
    promoted = validation_passed and config.auto_promote
    if promoted:
        client.set_registered_model_alias(
            config.registered_model_name,
            config.model_alias,
            str(model_version),
        )

    return ProductionTrainingResult(
        run_id=run_id,
        model_version=int(model_version),
        metrics=metrics,
        validation_passed=validation_passed,
        promoted=promoted,
    )
