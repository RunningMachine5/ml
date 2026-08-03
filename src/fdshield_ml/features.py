"""FDShield 공개 데이터의 검증, 라벨 변환, Feature 생성을 담당하는 모듈.

학습 데이터와 실시간 추론 입력에 동일한 변환을 적용할 수 있도록 Feature 처리를
``sklearn`` Transformer로 구현한다. 개인정보성 식별자는 모델 입력에서 제외하고,
계좌번호는 데이터 누수를 막기 위한 그룹 분할에만 사용한다.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET_COLUMN = "Fraud_Type"
GROUP_COLUMN = "Account_account_number"
NORMAL_LABEL = "m"
FRAUD_LABELS = frozenset("abcdefghijkl")

# 직접 식별자와 고유값이 지나치게 많은 값은 학습 입력에서 제외합니다.
# Account_account_number는 데이터 분할 그룹으로만 사용합니다.
EXCLUDED_FEATURE_COLUMNS = (
    "ID",
    "Customer_personal_identifier",
    "Customer_identification_number",
    GROUP_COLUMN,
    "IP_Address",
    "MAC_Address",
    "Location",
    "Recipient_Account_Number",
)

DATETIME_COLUMNS = (
    "Customer_registration_datetime",
    "Account_creation_datetime",
    "Transaction_Datetime",
    "Last_atm_transaction_datetime",
    "Last_bank_branch_transaction_datetime",
    "Transaction_resumed_date",
)


def binary_target(values: pd.Series) -> pd.Series:
    """원본 라벨 ``m``은 정상(0), ``a``~``l``은 사기(1)로 변환한다."""

    # 공백과 대소문자 차이를 정리한 뒤 명세에 없는 값이 섞였는지 검사한다.
    labels = values.astype("string").str.strip().str.lower()
    allowed = FRAUD_LABELS | {NORMAL_LABEL}
    unexpected = sorted(set(labels.dropna().unique()) - allowed)
    if labels.isna().any() or unexpected:
        details = []
        if labels.isna().any():
            details.append(f"missing={int(labels.isna().sum())}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(f"Invalid {TARGET_COLUMN} labels: {', '.join(details)}")
    return labels.ne(NORMAL_LABEL).astype("int8").rename("is_fraud")


def model_input_and_groups(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """식별자를 제외한 모델 입력, 이진 라벨, 계좌 그룹을 분리해 반환한다."""

    missing = {TARGET_COLUMN, GROUP_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    # 계좌번호는 Feature로 학습하지 않고 그룹 분할용 Series로만 보존한다.
    target = binary_target(frame[TARGET_COLUMN])
    groups = frame[GROUP_COLUMN].astype("string")
    if groups.isna().any():
        # 결측 계좌들이 하나의 거대한 그룹이 되지 않도록 각 행에 별도 키를 줍니다.
        fallback = pd.Series(
            [f"__missing_account_{index}" for index in frame.index],
            index=frame.index,
            dtype="string",
        )
        groups = groups.fillna(fallback)

    # 정답 라벨과 직접 식별자를 모델 입력에서 제거하여 누수와 개인정보 학습을 막는다.
    excluded = [TARGET_COLUMN, *EXCLUDED_FEATURE_COLUMNS]
    model_input = frame.drop(columns=excluded, errors="ignore").copy()
    if model_input.empty:
        raise ValueError("No model features remain after excluding identifiers.")
    # MLflow 모델 시그니처에서 결측 가능한 정수 열이 거부되지 않도록 수치 입력을
    # 일관되게 float64로 기록합니다. JSON 정수 입력도 double 스키마에 들어옵니다.
    numeric_columns = model_input.select_dtypes(include=["number", "bool"]).columns
    model_input[numeric_columns] = model_input[numeric_columns].astype("float64")
    return model_input, target, groups.rename("account_group")


def _elapsed_hours(
    parsed: dict[str, pd.Series], later: str, earlier: str
) -> pd.Series | None:
    if later not in parsed or earlier not in parsed:
        return None
    return (parsed[later] - parsed[earlier]).dt.total_seconds() / 3600.0


class FDShieldFeatureBuilder(TransformerMixin, BaseEstimator):
    """식별자를 제외한 원시 컬럼을 학습 가능한 Feature로 변환한다."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FDShieldFeatureBuilder":
        self._validate_frame(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """날짜·시간 컬럼을 수치형 파생 Feature로 변환한다."""

        self._validate_frame(X)
        result = X.copy()

        # 파싱할 수 없는 날짜는 NaT로 두고 뒤의 Imputer가 처리하게 한다.
        parsed = {
            column: pd.to_datetime(result[column], errors="coerce")
            for column in DATETIME_COLUMNS
            if column in result.columns
        }

        # 거래 시각에서 월, 요일, 시간대처럼 모델이 사용할 수 있는 수치를 만든다.
        transaction_time = parsed.get("Transaction_Datetime")
        if transaction_time is not None:
            result["Transaction_month"] = transaction_time.dt.month
            result["Transaction_day"] = transaction_time.dt.day
            result["Transaction_hour"] = transaction_time.dt.hour
            result["Transaction_dayofweek"] = transaction_time.dt.dayofweek
            result["Transaction_is_weekend"] = (
                transaction_time.dt.dayofweek >= 5
            ).astype("float64")

        # 기준 시각 두 개의 차이로 고객/계좌 상태와 최근 활동 간격을 표현한다.
        elapsed_features = {
            "Customer_tenure_hours": (
                "Transaction_Datetime",
                "Customer_registration_datetime",
            ),
            "Account_age_hours": (
                "Transaction_Datetime",
                "Account_creation_datetime",
            ),
            "Hours_since_last_atm": (
                "Transaction_Datetime",
                "Last_atm_transaction_datetime",
            ),
            "Hours_since_last_branch": (
                "Transaction_Datetime",
                "Last_bank_branch_transaction_datetime",
            ),
            "Resume_delay_hours": (
                "Transaction_resumed_date",
                "Transaction_Datetime",
            ),
        }
        for feature_name, (later, earlier) in elapsed_features.items():
            elapsed = _elapsed_hours(parsed, later, earlier)
            if elapsed is not None:
                result[feature_name] = elapsed

        # 문자열 timedelta는 초 단위 수치로 바꿔 XGBoost가 처리할 수 있게 한다.
        if "Time_difference" in result.columns:
            result["Time_difference_seconds"] = pd.to_timedelta(
                result["Time_difference"], errors="coerce"
            ).dt.total_seconds()
            result = result.drop(columns=["Time_difference"])

        # 변환을 마친 원본 날짜 문자열은 중복 학습되지 않도록 제거한다.
        result = result.drop(columns=list(parsed), errors="ignore")
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("FDShieldFeatureBuilder expects a pandas DataFrame.")


def feature_manifest(columns: Iterable[str]) -> dict[str, object]:
    """원본 행 값 없이 모델의 Feature 구성만 설명하는 메타데이터를 만든다."""

    return {
        "target_mapping": {NORMAL_LABEL: 0, "a-l": 1},
        "group_split_column": GROUP_COLUMN,
        "excluded_identifier_columns": list(EXCLUDED_FEATURE_COLUMNS),
        "model_input_columns": list(columns),
    }
