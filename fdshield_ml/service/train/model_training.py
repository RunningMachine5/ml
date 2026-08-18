"""
[이진 탐색기 학습 코드]
전처리가 끝난 model79 데이터를 학습용과 검증용으로 나누고 XGBoost 모델을
학습한다. GCS, MLflow, Backend Callback 같은 운영 연결은 이 파일에서 다루지
않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from fdshield_ml.service.xgboost_prediction import prediction_iteration_range

DECISION_THRESHOLD = 0.5
VALIDATION_FRACTION = 0.2


class ModelTrainingError(RuntimeError):
    """학습 데이터 분할 또는 모델 학습이 실패했을 때 발생한다."""


@dataclass(frozen=True)
class ModelTrainingConfig:
    """doo 최신 main에서 사용하는 XGBoost 학습 설정."""

    random_state: int = 42
    n_estimators: int = 10_000
    max_depth: int = 6
    learning_rate: float = 0.01
    min_child_weight: float = 1.0
    gamma: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    scale_pos_weight: float = 65.1
    tree_method: str = "hist"
    early_stopping_rounds: int = 1_000
    n_jobs: int = -1


@dataclass(frozen=True)
class ModelTrainingResult:
    """학습된 후보 모델과 동일 검증 세트의 평가 결과.

    검증 데이터도 반환하는 이유는 MLflow에서 현재 운영 모델과 후보 모델을
    완전히 같은 데이터로 비교하기 위해서다.
    """

    model: XGBClassifier
    validation_features: pd.DataFrame
    validation_target: pd.Series
    metrics: dict[str, float]
    decision_threshold: float
    train_rows: int
    validation_rows: int
    iteration_range: tuple[int, int] | None


def evaluation_metrics(
    target: pd.Series,
    probability: pd.Series,
) -> dict[str, float]:
    """후보와 현재 운영 모델을 고정 임계값 0.5로 평가한다."""

    # 학습·비교·실시간 판정 기준이 달라지지 않도록 doo 정책인 0.5를 공통 사용한다.
    predicted = probability.ge(DECISION_THRESHOLD).astype("int8")
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
        "decision_threshold": DECISION_THRESHOLD,
    }


def build_classifier(config: ModelTrainingConfig) -> XGBClassifier:
    """전달본과 동일한 XGBoost 분류기를 만든다."""

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        gamma=config.gamma,
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        scale_pos_weight=config.scale_pos_weight,
        tree_method=config.tree_method,
        enable_categorical=True,
        eval_metric="logloss",
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )


def train_model(
    train_data: pd.DataFrame,
    label: pd.Series,
    config: ModelTrainingConfig | None = None,
) -> ModelTrainingResult:
    """model79 데이터와 라벨을 분할해 XGBoost 후보 모델을 학습한다."""

    settings = config or ModelTrainingConfig()

    # 데이터 학습용, 검증용으로 분리
    try:
        x_train, x_valid, y_train, y_valid = train_test_split(
            train_data,
            label,
            test_size=VALIDATION_FRACTION,
            stratify=label,
            random_state=settings.random_state,
        )
    except ValueError as exc:
        raise ModelTrainingError(
            "train1 data cannot produce a stratified 80/20 split"
        ) from exc

    # 모델 정의, 학습
    model = build_classifier(settings)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=False,
    )
    # 검증 데이터의 예측 확률로 성능을 평가한다.
    # 조기 종료 뒤의 불필요한 트리는 평가에서 제외해 Serving 예측과 맞춘다.
    iteration_range = prediction_iteration_range(model, model.get_booster())
    probability = pd.Series(
        model.predict_proba(x_valid, iteration_range=iteration_range)[:, 1],
        index=y_valid.index,
    )
    return ModelTrainingResult(
        model=model,
        validation_features=x_valid,
        validation_target=y_valid,
        metrics=evaluation_metrics(y_valid, probability),
        decision_threshold=DECISION_THRESHOLD,
        train_rows=len(x_train),
        validation_rows=len(x_valid),
        iteration_range=iteration_range,
    )
