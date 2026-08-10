"""Optuna로 하이퍼파라미터를 탐색하고 모든 결과를 MLflow에 기록한다.

학습은 팀원 PC에서 실행된다. 공용 MLflow 서버는 Trial마다 사용한 파라미터와
검증지표를 받아 비교하고, 탐색이 끝나면 Best Trial의 모델 artifact만 저장한다.
Trial마다 큰 모델 파일을 업로드하지 않아 서버 디스크 낭비를 줄인다.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from mlflow.models import infer_signature

from fdshield_ml.common.decision_threshold import store_model_decision_threshold
from fdshield_ml.common.features import (
    FDShieldFeatureBuilder,
    feature_manifest,
    model_input_and_groups,
)
from fdshield_ml.training.pipeline import (
    TrainingConfig,
    build_pipeline,
    class_balance_weight,
    evaluate_pipeline,
    group_train_validation_split,
    model_parameters,
    stratified_sample,
)
from fdshield_ml.training.tracking import configure_tracking, verify_connection
from fdshield_ml.training.tuning import (
    config_from_best_params,
    suggest_training_config,
)

MODEL_TYPES = (
    "logistic-regression",
    "decision-tree",
    "random-forest",
    "xgboost",
)


def parse_args() -> argparse.Namespace:
    """팀원이 코드를 수정하지 않고 Study를 실행하도록 CLI를 정의한다."""

    parser = argparse.ArgumentParser(
        description=(
            "Tune an FDShield classifier with Optuna and log nested Trial runs "
            "plus the best model to MLflow."
        )
    )
    parser.add_argument("--data-path", default="data/open/train.csv")
    parser.add_argument("--env-file", default=".env.tracking")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument(
        "--comparison-group",
        default=None,
        help="Tag used to group studies that share the same comparison conditions.",
    )
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="xgboost")
    parser.add_argument("--registered-model-name", default=None)

    # 먼저 작은 표본과 적은 Trial로 전체 흐름을 확인하는 데 사용한다.
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum total tuning time in seconds. Omit for no time limit.",
    )

    # 모든 Trial이 같은 데이터 분할을 사용해야 파라미터만 공정하게 비교할 수 있다.
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 4))
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """긴 자동 탐색을 시작하기 전에 잘못된 옵션을 차단한다."""

    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1.")
    if args.timeout is not None and args.timeout < 1:
        raise ValueError("--timeout must be a positive number of seconds.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.n_jobs < 1 or args.logistic_max_iter < 1:
        raise ValueError("Job and iteration counts must be positive.")


def _trial_summary(study: optuna.Study) -> list[dict[str, object]]:
    """MLflow artifact로 저장할 JSON 직렬화 가능한 Trial 요약을 만든다."""

    summary: list[dict[str, object]] = []
    for trial in study.trials:
        summary.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "parameters": trial.params,
                "duration_seconds": (
                    trial.duration.total_seconds() if trial.duration else None
                ),
                "mlflow_run_id": trial.user_attrs.get("mlflow_run_id"),
            }
        )
    return summary


def main() -> None:
    """데이터 준비, Optuna Study, Best 모델 기록을 순서대로 실행한다."""

    args = parse_args()
    _validate_args(args)
    tracking_uri = configure_tracking(args.env_file)
    verify_connection()

    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {data_path}")

    raw_frame = pd.read_csv(data_path, low_memory=False)
    model_input, target, groups = model_input_and_groups(raw_frame)
    del raw_frame

    model_input, target = stratified_sample(
        model_input, target, args.max_rows, args.random_state
    )
    groups = groups.loc[model_input.index]

    base_config = TrainingConfig(
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state,
        logistic_max_iter=args.logistic_max_iter,
        n_jobs=args.n_jobs,
    )
    split = group_train_validation_split(
        model_input, target, groups, base_config
    )
    balance_weight = class_balance_weight(split.y_train)

    experiment_name = args.experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "fdshield-model-comparison"
    )
    study_name = args.study_name or f"optuna-{args.model_type}"
    mlflow.set_experiment(experiment_name)

    # Seed를 고정하면 같은 데이터와 Trial 수에서 탐색 순서를 재현하기 쉽다.
    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    # 부모 Run은 Study 전체를 나타내고 각 Trial Run은 그 아래에 중첩된다.
    with mlflow.start_run(run_name=study_name) as parent_run:
        mlflow.log_params(
            {
                "tuning_library": f"optuna-{optuna.__version__}",
                "objective_metric": "validation_pr_auc",
                "model_type": args.model_type,
                "requested_trials": args.n_trials,
                "timeout_seconds": args.timeout or "none",
                "source_file": data_path.name,
                "sampled_rows": len(model_input),
                "train_rows": len(split.X_train),
                "validation_rows": len(split.X_validation),
                "test_size": base_config.test_size,
                "random_state": base_config.random_state,
                "n_jobs_per_model": base_config.n_jobs,
            }
        )
        study_tags = {
            "project": "fdshield",
            "task": "binary_fraud_detection",
            "owner": os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"),
            "run_kind": "optuna_study",
            "model_family": args.model_type,
            "data_privacy": "direct_identifiers_excluded",
        }
        if args.comparison_group:
            study_tags["comparison_group"] = args.comparison_group
        mlflow.set_tags(study_tags)

        def objective(trial: optuna.Trial) -> float:
            """한 파라미터 조합을 학습하고 PR-AUC를 Optuna에 반환한다."""

            config = suggest_training_config(trial, base_config)
            pipeline, _, _ = build_pipeline(
                split.X_train, balance_weight, config
            )
            started = time.perf_counter()
            pipeline.fit(split.X_train, split.y_train)
            training_seconds = time.perf_counter() - started
            evaluation = evaluate_pipeline(
                pipeline, split.X_validation, split.y_validation
            )

            # Trial에는 비교에 필요한 숫자만 남기고 큰 모델은 업로드하지 않는다.
            with mlflow.start_run(
                run_name=f"{study_name}-trial-{trial.number:03d}",
                nested=True,
            ) as child_run:
                mlflow.log_params(
                    {
                        "trial_number": trial.number,
                        **model_parameters(config, balance_weight),
                    }
                )
                mlflow.log_metrics(
                    {
                        **evaluation.metrics,
                        "training_seconds": training_seconds,
                    }
                )
                trial_tags = {
                    "project": "fdshield",
                    "owner": os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"),
                    "run_kind": "optuna_trial",
                    "model_family": args.model_type,
                }
                if args.comparison_group:
                    trial_tags["comparison_group"] = args.comparison_group
                mlflow.set_tags(trial_tags)
                trial.set_user_attr("mlflow_run_id", child_run.info.run_id)

            return evaluation.metrics["validation_pr_auc"]

        # 모델 자체가 CPU 병렬화를 사용하므로 Trial은 한 번에 하나씩 실행한다.
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=1,
            gc_after_trial=True,
        )

        best_config = config_from_best_params(
            base_config, study.best_trial.params
        )
        best_pipeline, numeric_columns, categorical_columns = build_pipeline(
            split.X_train, balance_weight, best_config
        )
        best_started = time.perf_counter()
        best_pipeline.fit(split.X_train, split.y_train)
        best_training_seconds = time.perf_counter() - best_started
        best_evaluation = evaluate_pipeline(
            best_pipeline, split.X_validation, split.y_validation
        )

        # Parent Run은 Best Trial의 선택 근거와 실제 배포 후보 모델을 함께 보관한다.
        mlflow.log_params(
            {
                "best_trial_number": study.best_trial.number,
                **{
                    f"best_{name}": value
                    for name, value in model_parameters(
                        best_config, balance_weight
                    ).items()
                    if name not in {"model_type", "random_state"}
                },
            }
        )
        mlflow.log_metrics(
            {
                **best_evaluation.metrics,
                "training_seconds": best_training_seconds,
                "completed_trials": float(len(study.trials)),
            }
        )
        mlflow.log_dict(
            {
                "study_name": study.study_name,
                "direction": study.direction.name,
                "objective_metric": "validation_pr_auc",
                "best_trial_number": study.best_trial.number,
                "best_value": study.best_value,
                "best_parameters": study.best_trial.params,
                "trials": _trial_summary(study),
            },
            "tuning/study_summary.json",
        )
        mlflow.log_dict(
            {
                "metrics": best_evaluation.metrics,
                "confusion_matrix": best_evaluation.confusion_matrix,
                "classification_report": best_evaluation.classification_report,
            },
            "evaluation/best_model_summary.json",
        )
        mlflow.log_dict(
            {
                **feature_manifest(model_input.columns),
                "engineered_numeric_columns": numeric_columns,
                "engineered_categorical_columns": categorical_columns,
            },
            "metadata/feature_manifest.json",
        )

        # 데이터 전체를 artifact로 올리지 않고 Dataset 스키마와 digest만 기록한다.
        feature_builder = FDShieldFeatureBuilder()
        validation_dataset_frame = feature_builder.transform(
            split.X_validation
        ).assign(is_fraud=split.y_validation.to_numpy())
        mlflow.log_input(
            mlflow.data.from_pandas(
                validation_dataset_frame,
                targets="is_fraud",
                name="fdshield_open_validation",
            ),
            context="validation",
        )

        input_example = split.X_validation.head(5).copy()
        predicted_example = best_pipeline.predict_proba(input_example)
        store_model_decision_threshold(
            best_pipeline,
            best_evaluation.metrics["decision_threshold"],
        )
        signature = infer_signature(input_example, predicted_example)
        model_info = mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
            pyfunc_predict_fn="predict_proba",
            registered_model_name=args.registered_model_name,
        )

        print(
            "Optuna study logged: "
            f"{tracking_uri}/#/experiments/{parent_run.info.experiment_id}"
            f"/runs/{parent_run.info.run_id}"
        )
        print(f"Study: {study.study_name}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best parameters: {study.best_trial.params}")
        print(f"Best validation PR-AUC: {study.best_value:.4f}")
        print(f"Logged best model: {model_info.model_uri}")


if __name__ == "__main__":
    main()
