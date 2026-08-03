"""MLflow 입출력과 분리하여 재사용하는 모델 학습·평가 함수.

이 모듈은 데이터 분할, 전처리 Pipeline 생성, 학습 결과 평가만 담당한다. MLflow
서버 연결과 기록을 섞지 않았기 때문에 단위 테스트에서 독립적으로 실행할 수 있고,
여러 분류 모델이 정확히 같은 데이터와 평가 방법을 재사용할 수 있다.

처음 보는 팀원은 ``build_pipeline``을 먼저 보면 된다. 앞부분의 Feature 생성과
전처리는 모든 모델이 공유하고, 마지막 ``classifier`` 단계만 ``model_type``에 따라
바뀐다. 따라서 모델을 비교할 때 이 파일을 복사하거나 전처리를 각자 수정하지
않아도 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from fdshield_ml.features import FDShieldFeatureBuilder


@dataclass(frozen=True)
class TrainingConfig:
    """한 번의 학습에서 사용할 공통 설정과 모델별 하이퍼파라미터.

    한 모델이 사용하지 않는 값도 이 객체 안에 있을 수 있다. 예를 들어
    ``logistic_c``는 Logistic Regression만 사용한다. MLflow에는 실제 선택된
    모델이 사용하는 값만 기록한다.
    """

    model_type: str = "xgboost"
    test_size: float = 0.2
    random_state: int = 42

    # XGBoost와 Random Forest가 사용하는 트리 개수다.
    n_estimators: int = 300

    # XGBoost, Decision Tree, Random Forest가 사용하는 트리 깊이다.
    max_depth: int = 5

    # 아래 네 값은 XGBoost에서만 사용한다.
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0

    # Decision Tree와 Random Forest가 사용하는 리프의 최소 데이터 수다.
    min_samples_leaf: int = 5
    # Random Forest가 각 분기에서 검토할 Feature의 범위다.
    max_features: str = "sqrt"

    # Logistic Regression의 규제 강도와 최대 반복 횟수다.
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000

    n_jobs: int = 4
    min_category_frequency: int = 5


@dataclass(frozen=True)
class SplitData:
    """계좌 단위로 분리된 학습/검증 데이터와 그룹 정보."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    train_groups: pd.Series
    validation_groups: pd.Series


@dataclass(frozen=True)
class EvaluationResult:
    """MLflow와 콘솔에 기록할 검증 결과 모음."""

    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    classification_report: dict[str, object]
    probabilities: np.ndarray


def stratified_sample(
    frame: pd.DataFrame,
    target: pd.Series,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """클래스 비율을 최대한 유지하는 재현 가능한 Smoke Test 표본을 만든다."""

    if max_rows is None or max_rows >= len(frame):
        return frame, target
    if max_rows < 200:
        raise ValueError("--max-rows must be at least 200 for this 99:1 dataset.")

    # 정상/사기별로 따로 추출하여 1%뿐인 사기 행이 표본에서 사라지지 않게 한다.
    sampled_indices: list[object] = []
    ratio = max_rows / len(frame)
    for label, label_indices in target.groupby(target).groups.items():
        sample_count = max(1, round(len(label_indices) * ratio))
        sample_count = min(sample_count, len(label_indices))
        sampled_indices.extend(
            pd.Series(list(label_indices)).sample(
                n=sample_count,
                random_state=random_state + int(label),
            )
        )

    # 반올림 때문에 목표 행 수와 차이가 나면 남은 행을 추가하거나 일부를 줄인다.
    if len(sampled_indices) < max_rows:
        remaining = frame.index.difference(pd.Index(sampled_indices))
        extra_count = min(max_rows - len(sampled_indices), len(remaining))
        sampled_indices.extend(
            pd.Series(list(remaining)).sample(
                n=extra_count, random_state=random_state + 100
            )
        )
    elif len(sampled_indices) > max_rows:
        sampled_indices = list(
            pd.Series(sampled_indices).sample(n=max_rows, random_state=random_state)
        )

    sampled_index = pd.Index(sampled_indices)
    return frame.loc[sampled_index].copy(), target.loc[sampled_index].copy()


def group_train_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: TrainingConfig,
) -> SplitData:
    """동일 계좌가 학습과 검증에 동시에 들어가지 않도록 그룹 분할한다."""

    # 99:1 데이터에서는 한 번의 분할로 검증 세트에 사기 행이 없을 수 있으므로
    # random_state를 바꿔 가며 양쪽에 두 클래스가 모두 포함되는 분할을 찾는다.
    for attempt in range(20):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=config.test_size,
            random_state=config.random_state + attempt,
        )
        train_index, validation_index = next(splitter.split(X, y, groups))
        y_train = y.iloc[train_index]
        y_validation = y.iloc[validation_index]
        if y_train.nunique() == 2 and y_validation.nunique() == 2:
            split = SplitData(
                X_train=X.iloc[train_index].copy(),
                X_validation=X.iloc[validation_index].copy(),
                y_train=y_train.copy(),
                y_validation=y_validation.copy(),
                train_groups=groups.iloc[train_index].copy(),
                validation_groups=groups.iloc[validation_index].copy(),
            )
            # 이후 코드가 바뀌더라도 계좌 누수를 즉시 발견하도록 방어적으로 검사한다.
            overlap = set(split.train_groups) & set(split.validation_groups)
            if overlap:
                raise RuntimeError("Account groups leaked across the split.")
            return split
    raise ValueError(
        "Could not create train/validation groups containing both classes. "
        "Use more rows or a different random seed."
    )


def class_balance_weight(target: pd.Series) -> float:
    """정상/사기 개수 비율로 XGBoost의 ``scale_pos_weight``를 계산한다."""

    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training data must contain both normal and fraud rows.")
    return negatives / positives


def model_parameters(
    config: TrainingConfig, scale_pos_weight: float
) -> dict[str, object]:
    """MLflow에 기록할 선택 모델의 하이퍼파라미터만 반환한다.

    모델과 관계없는 옵션까지 MLflow에 기록하면 초보자가 어떤 값이 실제 학습에
    사용됐는지 혼동하기 쉽다. 이 함수는 예를 들어 Logistic Regression Run에는
    ``max_depth``가 나타나지 않도록 모델별 값을 명확히 나눈다.
    """

    common: dict[str, object] = {
        "model_type": config.model_type,
        "random_state": config.random_state,
    }
    if config.model_type == "logistic-regression":
        return {
            **common,
            "C": config.logistic_c,
            "max_iter": config.logistic_max_iter,
            "class_weight": "balanced",
            "solver": "liblinear",
        }
    if config.model_type == "decision-tree":
        return {
            **common,
            "max_depth": config.max_depth,
            "min_samples_leaf": config.min_samples_leaf,
            "class_weight": "balanced",
        }
    if config.model_type == "random-forest":
        return {
            **common,
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "min_samples_leaf": config.min_samples_leaf,
            "max_features": config.max_features,
            "class_weight": "balanced_subsample",
            "n_jobs": config.n_jobs,
        }
    if config.model_type == "xgboost":
        return {
            **common,
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "min_child_weight": config.min_child_weight,
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": config.n_jobs,
        }
    raise ValueError(f"Unsupported model type: {config.model_type}")


def build_classifier(config: TrainingConfig, scale_pos_weight: float) -> object:
    """문자열로 선택한 분류 모델을 생성한다.

    네 모델 모두 ``predict_proba``를 지원하기 때문에 뒤의 PR-AUC, Recall, F1
    계산 코드는 모델 종류와 상관없이 그대로 재사용할 수 있다.

    - Logistic Regression: 복잡한 모델과 비교할 가장 단순한 선형 기준점
    - Decision Tree: 한 개 트리의 규칙을 이해하기 쉬운 기준점
    - Random Forest: 여러 트리를 평균내 과적합을 줄인 앙상블 기준점
    - XGBoost: 이전 트리의 오차를 다음 트리가 보완하는 주력 후보
    """

    if config.model_type == "logistic-regression":
        return LogisticRegression(
            C=config.logistic_c,
            max_iter=config.logistic_max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=config.random_state,
        )
    if config.model_type == "decision-tree":
        return DecisionTreeClassifier(
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced",
            random_state=config.random_state,
        )
    if config.model_type == "random-forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            class_weight="balanced_subsample",
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
    if config.model_type == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_weight=config.min_child_weight,
            scale_pos_weight=scale_pos_weight,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
    raise ValueError(f"Unsupported model type: {config.model_type}")


def build_pipeline(
    raw_training_features: pd.DataFrame,
    scale_pos_weight: float,
    config: TrainingConfig,
) -> tuple[Pipeline, list[str], list[str]]:
    """원시 입력부터 사기 확률까지 처리하는 단일 sklearn Pipeline을 만든다."""

    # Pipeline에 Feature Builder를 포함해야 학습과 추론이 같은 변환을 사용한다.
    feature_builder = FDShieldFeatureBuilder()
    engineered = feature_builder.fit_transform(raw_training_features)

    # 수치형과 범주형은 결측치 처리 방법이 달라 별도의 하위 Pipeline으로 구성한다.
    numeric_columns = list(engineered.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [
        column for column in engineered.columns if column not in numeric_columns
    ]

    # 수치형 결측치는 이상치의 영향을 덜 받도록 중앙값으로 대체한다.
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    # Logistic Regression은 금액처럼 단위가 큰 Feature에 지나치게 끌릴 수 있어
    # 수치 범위를 맞춘다. 트리 계열 모델은 값의 순서로 분기하므로 스케일링이
    # 필요하지 않다. with_mean=False는 One-Hot 결과의 희소 행렬을 유지한다.
    if config.model_type == "logistic-regression":
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    # 처음 보는 범주도 추론할 수 있게 처리하고, 희귀 범주는 묶어 차원을 줄인다.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=config.min_category_frequency,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=1.0,
    )
    # 분류기만 교체하고 그 앞의 Feature/전처리/평가 과정은 모든 모델이 공유한다.
    classifier = build_classifier(config, scale_pos_weight)
    # 저장되는 모델 하나에 Feature 생성, 전처리, 분류기를 모두 포함한다.
    pipeline = Pipeline(
        steps=[
            ("feature_builder", feature_builder),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline, numeric_columns, categorical_columns


def best_f1_threshold(target: pd.Series, probabilities: np.ndarray) -> float:
    """검증 데이터에서 F1이 가장 높은 사기 판정 임계값을 찾는다."""

    # predict_proba의 기본 0.5를 고정하지 않고 Precision-Recall 곡선을 탐색한다.
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    if len(thresholds) == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def evaluate_pipeline(
    pipeline: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> EvaluationResult:
    """검증 확률을 판정값으로 바꾸고 불균형 분류용 지표를 계산한다."""

    probabilities = pipeline.predict_proba(X_validation)[:, 1]
    threshold = best_f1_threshold(y_validation, probabilities)
    predictions = (probabilities >= threshold).astype("int8")

    # labels 순서를 고정하여 한 클래스가 적은 경우에도 행렬 의미가 바뀌지 않게 한다.
    matrix = confusion_matrix(y_validation, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    false_positive_rate = false_positive / max(false_positive + true_negative, 1)

    # Accuracy는 99% 정상 데이터에서 과대평가되므로 PR-AUC와 Recall 등을 함께 본다.
    metrics = {
        "validation_pr_auc": float(
            average_precision_score(y_validation, probabilities)
        ),
        "validation_roc_auc": float(roc_auc_score(y_validation, probabilities)),
        "validation_precision": float(
            precision_score(y_validation, predictions, zero_division=0)
        ),
        "validation_recall": float(
            recall_score(y_validation, predictions, zero_division=0)
        ),
        "validation_f1": float(f1_score(y_validation, predictions, zero_division=0)),
        "validation_false_positive_rate": float(false_positive_rate),
        "validation_accuracy": float(accuracy_score(y_validation, predictions)),
        "decision_threshold": threshold,
        "validation_true_negative": float(true_negative),
        "validation_false_positive": float(false_positive),
        "validation_false_negative": float(false_negative),
        "validation_true_positive": float(true_positive),
    }
    report = classification_report(
        y_validation,
        predictions,
        labels=[0, 1],
        target_names=["normal", "fraud"],
        output_dict=True,
        zero_division=0,
    )
    return EvaluationResult(
        metrics=metrics,
        confusion_matrix=matrix.astype(int).tolist(),
        classification_report=report,
        probabilities=probabilities,
    )
