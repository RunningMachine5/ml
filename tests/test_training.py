from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from fdshield_ml.features import (
    EXCLUDED_FEATURE_COLUMNS,
    FDShieldFeatureBuilder,
    binary_target,
    model_input_and_groups,
)
from fdshield_ml.training import (
    TrainingConfig,
    build_pipeline,
    class_balance_weight,
    evaluate_pipeline,
    group_train_validation_split,
    model_parameters,
)


def synthetic_frame(group_count: int = 40, rows_per_group: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for group in range(group_count):
        for row_number in range(rows_per_group):
            is_fraud = row_number == 0
            transaction_time = pd.Timestamp("2025-01-01") + timedelta(
                hours=group * rows_per_group + row_number
            )
            rows.append(
                {
                    "ID": f"TRAIN_{group}_{row_number}",
                    "Customer_personal_identifier": f"person-{group}",
                    "Customer_identification_number": f"secret-{group}",
                    "Account_account_number": f"account-{group}",
                    "IP_Address": f"10.0.0.{group % 255}",
                    "MAC_Address": f"00:00:00:00:00:{group % 99:02d}",
                    "Location": "redacted location",
                    "Recipient_Account_Number": f"recipient-{group}",
                    "Customer_Birthyear": 1970 + group % 30,
                    "Customer_Gender": "male" if group % 2 else "female",
                    "Customer_credit_rating": ["A", "B", "C"][group % 3],
                    "Transaction_Datetime": str(transaction_time),
                    "Account_creation_datetime": str(
                        transaction_time - timedelta(days=365)
                    ),
                    "Customer_registration_datetime": str(
                        transaction_time - timedelta(days=730)
                    ),
                    "Transaction_Amount": float(
                        rng.normal(2_000_000 if is_fraud else 20_000, 1_000)
                    ),
                    "Channel": "mobile" if group % 2 else "internet",
                    "Time_difference": (
                        "0 days 00:00:10" if is_fraud else "0 days 02:00:00"
                    ),
                    "Fraud_Type": "a" if is_fraud else "m",
                }
            )
    return pd.DataFrame(rows)


def test_binary_target_maps_m_to_normal_and_a_to_l_to_fraud() -> None:
    values = pd.Series(["m", "a", "f", "l"])
    assert binary_target(values).tolist() == [0, 1, 1, 1]


def test_model_input_excludes_identifiers_and_builds_datetime_features() -> None:
    frame = synthetic_frame(group_count=2)
    X, target, groups = model_input_and_groups(frame)

    assert target.sum() == 2
    assert groups.nunique() == 2
    assert set(EXCLUDED_FEATURE_COLUMNS).isdisjoint(X.columns)

    engineered = FDShieldFeatureBuilder().fit_transform(X)
    assert "Transaction_Datetime" not in engineered.columns
    assert "Transaction_hour" in engineered.columns
    assert "Account_age_hours" in engineered.columns
    assert "Time_difference_seconds" in engineered.columns


@pytest.mark.parametrize(
    ("model_type", "expected_classifier"),
    [
        ("logistic-regression", "LogisticRegression"),
        ("decision-tree", "DecisionTreeClassifier"),
        ("random-forest", "RandomForestClassifier"),
        ("xgboost", "XGBClassifier"),
    ],
)
def test_each_model_uses_the_same_leak_free_training_flow(
    model_type: str, expected_classifier: str
) -> None:
    """네 모델이 동일한 계좌 분할·Feature·평가 흐름을 재사용하는지 확인한다."""

    frame = synthetic_frame()
    X, target, groups = model_input_and_groups(frame)
    config = TrainingConfig(
        model_type=model_type,
        n_estimators=8,
        max_depth=2,
        logistic_max_iter=200,
        n_jobs=1,
    )
    split = group_train_validation_split(X, target, groups, config)

    assert set(split.train_groups).isdisjoint(set(split.validation_groups))

    pipeline, _, _ = build_pipeline(
        split.X_train,
        class_balance_weight(split.y_train),
        config,
    )
    classifier_name = pipeline.named_steps["classifier"].__class__.__name__
    assert classifier_name == expected_classifier

    pipeline.fit(split.X_train, split.y_train)
    result = evaluate_pipeline(
        pipeline, split.X_validation, split.y_validation
    )

    assert 0 <= result.metrics["validation_pr_auc"] <= 1
    assert 0 <= result.metrics["validation_recall"] <= 1
    assert len(result.confusion_matrix) == 2

    # MLflow 화면에는 선택 모델이 실제로 사용한 파라미터만 나타나야 한다.
    logged_params = model_parameters(config, class_balance_weight(split.y_train))
    assert logged_params["model_type"] == model_type
    if model_type == "logistic-regression":
        assert "C" in logged_params
        assert "n_estimators" not in logged_params
