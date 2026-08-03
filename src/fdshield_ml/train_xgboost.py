"""여러 이진분류 모델을 같은 조건으로 학습하고 MLflow에 기록하는 실행 코드.

이 파일은 FDShield의 최종 학습 파이프라인을 미리 확정한 코드가 아니다. 팀원이
로컬 CSV로 공통 전처리와 평가를 수행하고, Logistic Regression, Decision Tree,
Random Forest, XGBoost 결과를 원격 MLflow에서 비교하기 위한 베이스라인이다.

파일 이름은 기존 ``fdshield-train-xgboost`` 명령과의 호환성을 위해 유지한다.
팀원은 코드를 직접 수정하지 말고 ``fdshield-train --model-type ...`` 명령의 옵션만
바꿔 실행한다. 최종 모델이 결정되면 선택된 모델을 운영용 파이프라인으로 정리한다.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from fdshield_ml.features import (
    FDShieldFeatureBuilder,
    feature_manifest,
    model_input_and_groups,
)
from fdshield_ml.tracking import configure_tracking, verify_connection
from fdshield_ml.training import (
    TrainingConfig,
    build_pipeline,
    class_balance_weight,
    evaluate_pipeline,
    group_train_validation_split,
    model_parameters,
    stratified_sample,
)


MODEL_TYPES = (
    "logistic-regression",
    "decision-tree",
    "random-forest",
    "xgboost",
)


def parse_args() -> argparse.Namespace:
    """예제를 코드 수정 없이 반복 실행할 수 있도록 CLI 옵션을 정의한다."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a normal/fraud classifier locally and log the run to the "
            "shared MLflow server."
        )
    )

    # 데이터와 MLflow Run을 식별하는 기본 옵션이다.
    parser.add_argument("--data-path", default="data/open/train.csv")
    parser.add_argument("--env-file", default=".env.tracking")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--model-type",
        choices=MODEL_TYPES,
        default="xgboost",
        help="Classifier to train. The default remains xgboost for compatibility.",
    )

    # 일반 실험에서는 생략한다. 배포 후보만 Registry에 등록할 때 이름을 전달한다.
    parser.add_argument("--registered-model-name", default=None)

    # 전체 데이터 학습 전 일부 행으로 흐름을 확인하는 Smoke Test 옵션이다.
    parser.add_argument("--max-rows", type=int, default=None)

    # 데이터 분할 조건이다. 공정한 비교를 위해 팀 전체가 기본값을 고정한다.
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    # 트리 계열 모델이 공통으로 사용하는 옵션이다. n-estimators는 Random Forest와
    # XGBoost에, max-depth는 세 가지 트리 모델에 사용된다.
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", choices=("sqrt", "log2"), default="sqrt")

    # 아래 네 옵션은 XGBoost를 선택했을 때만 사용한다.
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=1.0)

    # 아래 두 옵션은 Logistic Regression을 선택했을 때만 사용한다.
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--logistic-max-iter", type=int, default=1000)

    parser.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 4))
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """잘못된 옵션 때문에 긴 학습이 낭비되지 않도록 실행 전에 검증한다."""

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.n_estimators < 1 or args.max_depth < 1 or args.n_jobs < 1:
        raise ValueError("Estimator, depth, and job counts must be positive.")
    if args.min_samples_leaf < 1 or args.logistic_max_iter < 1:
        raise ValueError(
            "Leaf size and Logistic Regression iterations must be positive."
        )
    if args.logistic_c <= 0 or args.min_child_weight <= 0:
        raise ValueError("Regularization and child weight values must be positive.")
    for name in ("learning_rate", "subsample", "colsample_bytree"):
        value = getattr(args, name)
        if value <= 0 or (name != "learning_rate" and value > 1):
            raise ValueError(f"Invalid --{name.replace('_', '-')}: {value}")


def main() -> None:
    """데이터 준비부터 MLflow 기록까지 선택 모델의 전체 흐름을 실행한다."""

    # 1. 옵션과 MLflow 접속 정보를 먼저 검사한다.
    args = parse_args()
    _validate_args(args)
    tracking_uri = configure_tracking(args.env_file)
    verify_connection()  # 비싼 학습을 시작하기 전에 계정/네트워크를 확인합니다.

    # 2. 원본 CSV는 팀원 PC에서만 읽는다. 원본 행은 MLflow에 업로드하지 않는다.
    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {data_path}")

    raw_frame = pd.read_csv(data_path, low_memory=False)

    # 라벨을 정상/사기로 바꾸고 식별자를 제외하며, 계좌번호는 분할 그룹으로만 쓴다.
    model_input, target, groups = model_input_and_groups(raw_frame)
    del raw_frame

    # --max-rows가 주어지면 99:1 비율을 최대한 유지한 작은 표본을 만든다.
    model_input, target = stratified_sample(
        model_input, target, args.max_rows, args.random_state
    )
    groups = groups.loc[model_input.index]

    # 3. 실행 옵션을 재사용 가능한 학습 설정 객체로 묶는다.
    config = TrainingConfig(
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        logistic_c=args.logistic_c,
        logistic_max_iter=args.logistic_max_iter,
        n_jobs=args.n_jobs,
    )
    # 같은 계좌가 학습과 검증에 동시에 들어가는 데이터 누수를 막는다.
    split = group_train_validation_split(model_input, target, groups, config)

    # 정상/사기 비율을 계산한다. XGBoost는 이 값을 scale_pos_weight로 사용하고,
    # sklearn 모델은 class_weight="balanced" 계열 설정으로 같은 문제를 보정한다.
    balance_weight = class_balance_weight(split.y_train)

    # Feature 생성, 결측치 처리, One-Hot Encoding, 선택 모델을 한 Pipeline으로 묶는다.
    pipeline, numeric_columns, categorical_columns = build_pipeline(
        split.X_train, balance_weight, config
    )

    # 4. 팀원 PC에서 모델을 학습하고 검증 데이터로 성능을 계산한다.
    training_started = time.perf_counter()
    pipeline.fit(split.X_train, split.y_train)
    training_seconds = time.perf_counter() - training_started
    evaluation = evaluate_pipeline(
        pipeline, split.X_validation, split.y_validation
    )

    # 5. 같은 목적의 Run들이 한 화면에 모이도록 Experiment를 선택한다.
    experiment_name = args.experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "fdshield-model-comparison"
    )
    mlflow.set_experiment(experiment_name)

    # start_run 블록 안에서 기록한 모든 값이 하나의 MLflow Run에 묶인다.
    with mlflow.start_run(run_name=args.run_name) as run:
        # 재현과 비교에 필요한 설정값을 파라미터로 기록한다.
        mlflow.log_params(
            {
                "target_mapping": "m=0,a-l=1",
                "split_strategy": "GroupShuffleSplit(Account_account_number)",
                "source_file": data_path.name,
                "sampled_rows": len(model_input),
                "train_rows": len(split.X_train),
                "validation_rows": len(split.X_validation),
                "train_fraud_rows": int(split.y_train.sum()),
                "validation_fraud_rows": int(split.y_validation.sum()),
                "test_size": config.test_size,
                "min_category_frequency": config.min_category_frequency,
                **model_parameters(config, balance_weight),
            }
        )
        # 성능값은 MLflow UI에서 Run별 표와 그래프로 비교할 수 있다.
        mlflow.log_metrics(
            {**evaluation.metrics, "training_seconds": training_seconds}
        )
        # 태그는 프로젝트나 작업 종류를 기준으로 Run을 검색할 때 사용한다.
        mlflow.set_tags(
            {
                "project": "fdshield",
                "task": "binary_fraud_detection",
                "model_family": config.model_type,
                "data_privacy": "direct_identifiers_excluded",
                "tracking_client": "fdshield-ml",
            }
        )

        # 6. 원본 행 대신 MLflow Dataset 메타데이터와 스키마를 기록한다.
        # from_pandas는 여기서 데이터셋을 설명하기 위한 객체를 만들며, 이 예제는
        # 거래별 원본 CSV를 별도 artifact 파일로 업로드하지 않는다.
        feature_builder = FDShieldFeatureBuilder()
        train_dataset_frame = feature_builder.transform(split.X_train).assign(
            is_fraud=split.y_train.to_numpy()
        )
        validation_dataset_frame = feature_builder.transform(
            split.X_validation
        ).assign(is_fraud=split.y_validation.to_numpy())
        mlflow.log_input(
            mlflow.data.from_pandas(
                train_dataset_frame,
                targets="is_fraud",
                name="fdshield_open_train",
            ),
            context="training",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                validation_dataset_frame,
                targets="is_fraud",
                name="fdshield_open_validation",
            ),
            context="validation",
        )

        # 7. 표 형태로 보기 어려운 상세 평가 결과는 JSON artifact로 남긴다.
        mlflow.log_dict(
            {
                "metrics": evaluation.metrics,
                "confusion_matrix": evaluation.confusion_matrix,
            },
            "evaluation/summary.json",
        )
        mlflow.log_dict(
            evaluation.classification_report,
            "evaluation/classification_report.json",
        )
        mlflow.log_dict(
            {
                **feature_manifest(model_input.columns),
                "engineered_numeric_columns": numeric_columns,
                "engineered_categorical_columns": categorical_columns,
            },
            "metadata/feature_manifest.json",
        )

        # 8. 전처리와 모델을 함께 저장해 추론 시 동일한 변환을 재사용한다.
        input_example = split.X_validation.head(5).copy()
        predicted_example = pipeline.predict_proba(input_example)

        # Signature는 모델 서버가 기대하는 입력/출력 스키마를 MLflow에 남긴다.
        signature = infer_signature(input_example, predicted_example)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
            pyfunc_predict_fn="predict_proba",
            # None이면 일반 Run artifact로만 저장하고, 이름이 있으면 Registry에도
            # 새 모델 버전을 만든다.
            registered_model_name=args.registered_model_name,
        )

        # 팀원이 바로 결과 화면을 열고 핵심 지표를 확인할 수 있도록 요약한다.
        print(
            "MLflow run logged: "
            f"{tracking_uri}/#/experiments/{run.info.experiment_id}"
            f"/runs/{run.info.run_id}"
        )
        print(f"Run ID: {run.info.run_id}")
        print(f"Logged model: {model_info.model_uri}")
        print(
            "Validation: "
            f"PR-AUC={evaluation.metrics['validation_pr_auc']:.4f}, "
            f"Recall={evaluation.metrics['validation_recall']:.4f}, "
            f"Precision={evaluation.metrics['validation_precision']:.4f}, "
            f"F1={evaluation.metrics['validation_f1']:.4f}"
        )

if __name__ == "__main__":
    # python -m fdshield_ml.train_xgboost 로 직접 실행할 때의 진입점이다.
    main()
