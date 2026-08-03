"""Optuna가 탐색할 모델별 하이퍼파라미터 공간.

수동 학습과 자동 튜닝이 서로 다른 모델 설정을 만들지 않도록, Optuna Trial이
제안한 값을 기존 :class:`TrainingConfig`에 반영하는 역할만 담당한다. 데이터
분할·Feature 처리·평가는 ``training.py``의 공통 함수를 그대로 사용한다.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import optuna

from fdshield_ml.training import TrainingConfig


def suggest_training_config(
    trial: optuna.Trial,
    base_config: TrainingConfig,
) -> TrainingConfig:
    """선택 모델에 필요한 값만 탐색하여 새 학습 설정을 반환한다.

    탐색 범위는 입문 프로젝트에서 지나치게 오래 걸리지 않으면서도 기본값 주변을
    충분히 비교하도록 제한했다. ``log=True``인 값은 작은 값과 큰 값을 같은
    비중으로 살펴보는 로그 스케일 탐색이다.
    """

    if base_config.model_type == "logistic-regression":
        parameters = {
            "logistic_c": trial.suggest_float(
                "logistic_c", 1e-3, 100.0, log=True
            ),
        }
    elif base_config.model_type == "decision-tree":
        parameters = {
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 50, log=True
            ),
        }
    elif base_config.model_type == "random-forest":
        parameters = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 500, step=50
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 30, log=True
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
        }
    elif base_config.model_type == "xgboost":
        parameters = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 600, step=50
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
        }
    else:
        raise ValueError(f"Unsupported model type: {base_config.model_type}")

    return replace(base_config, **parameters)


def config_from_best_params(
    base_config: TrainingConfig,
    parameters: Mapping[str, object],
) -> TrainingConfig:
    """완료된 Best Trial의 파라미터를 학습 설정으로 복원한다."""

    allowed_fields = set(TrainingConfig.__dataclass_fields__)
    unexpected = set(parameters) - allowed_fields
    if unexpected:
        raise ValueError(f"Unexpected tuning parameters: {sorted(unexpected)}")
    return replace(base_config, **dict(parameters))
