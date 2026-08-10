"""실제 모델 연결 전 API 흐름을 검증하는 결정적 거래 위험 예측기."""

import hashlib
import math
import os
from typing import Protocol

from fdshield_ml.common.preprocessing import preprocess_transaction_features
from fdshield_ml.serving.schemas import PredictionRequest, PredictionResponse


class Predictor(Protocol):
    """서빙 API가 의존하는 최소 예측 인터페이스."""

    model_name: str
    model_version: str
    ready: bool

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """거래 한 건의 사기 여부와 설명값을 반환한다."""


class StubPredictor:
    """실제 모델 연결 전 여러 거래 위험 신호를 조합하는 결정적 Stub.

    정답 라벨을 사용하지 않고 실제 54→91 전처리 결과만 읽는다. 반환하는
    ``shap``은 실제 모델의 SHAP가 아니라, 확률을 만든 log-odds 기여도를 실제
    XGBoost 응답과 비슷한 91개 Feature 형태로 가공한 설명용 더미 값이다.
    """

    # 생성형 transactions.csv에서 평범한 소액 거래는 낮게 유지하되, 고액·한도·
    # 단말/계좌 이상징후가 여러 개 겹치면 기본 임계값(0.55)을 넘도록 보정했다.
    BASE_LOG_ODDS = -3.2

    def __init__(
        self,
        *,
        threshold: float = 0.55,
        model_name: str = "fdshield-rule-based-stub",
        model_version: str = "0",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold는 0과 1 사이여야 합니다.")

        self.threshold = threshold
        self.model_name = model_name
        self.model_version = model_version
        self.ready = True

    @classmethod
    def from_environment(cls) -> "StubPredictor":
        """Stub 모델 메타데이터를 환경변수에서 읽고 판정 기준은 코드에 고정한다."""

        return cls(
            model_name=os.getenv(
                "ML_MODEL_NAME",
                "fdshield-rule-based-stub",
            ),
            model_version=os.getenv("ML_MODEL_VERSION", "").strip() or "0",
        )

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """54개 원본 값을 91개로 전처리한 뒤 거래 기반 위험도를 계산한다."""

        # 실제 모델이 연결되면 이 DataFrame을 그대로 predict_proba에 전달한다.
        model_features = preprocess_transaction_features(request.features)
        features = model_features.iloc[0]
        contributions = _stub_risk_contributions(features)
        log_odds = self.BASE_LOG_ODDS + math.fsum(contributions.values())
        probability = round(_sigmoid(log_odds), 4)
        shap = _stub_shap_contributions(features, contributions)

        return PredictionResponse(
            transaction_id=request.transaction_id,
            is_fraud=probability >= self.threshold,
            fraud_probability=probability,
            shap=shap,
            model_name=self.model_name,
            model_version=self.model_version,
        )


def _stub_risk_contributions(features: object) -> dict[str, float]:
    """전처리된 거래 한 건에서 설명 가능한 log-odds 기여도를 만든다."""

    contributions: dict[str, float] = {}

    def add(feature_name: str, value: float) -> None:
        if value:
            contributions[feature_name] = round(
                contributions.get(feature_name, 0.0) + value,
                4,
            )

    amount = abs(_as_float(features["Transaction_Amount"]))
    balance = max(abs(_as_float(features["Account_balance"])), 1.0)
    daily_limit = max(
        abs(_as_float(features["Account_amount_daily_limit"])),
        1.0,
    )
    monthly_max = max(
        abs(_as_float(features["Account_one_month_max_amount"])),
        0.0,
    )
    monthly_std = max(
        abs(_as_float(features["Account_one_month_std_dev"])),
        0.0,
    )

    # 절대 금액보다 고객의 잔액·한도·평소 거래 범위와의 상대적 차이를 더 본다.
    add(
        "Transaction_Amount",
        0.80
        if amount >= 10_000_000
        else 0.45
        if amount >= 5_000_000
        else 0.20
        if amount >= 1_000_000
        else 0.0,
    )
    daily_ratio = amount / daily_limit
    add(
        "Account_amount_daily_limit",
        1.20
        if daily_ratio >= 1.0
        else 0.55
        if daily_ratio >= 0.75
        else 0.25
        if daily_ratio >= 0.40
        else 0.0,
    )
    balance_ratio = amount / balance
    add(
        "Account_balance",
        0.80
        if balance_ratio >= 1.0
        else 0.40
        if balance_ratio >= 0.70
        else 0.20
        if balance_ratio >= 0.40
        else 0.0,
    )
    if monthly_max > 0:
        monthly_max_ratio = amount / monthly_max
        add(
            "Account_one_month_max_amount",
            0.60
            if monthly_max_ratio >= 2.0
            else 0.35
            if monthly_max_ratio >= 1.0
            else 0.0,
        )
    elif amount >= 1_000_000:
        add("Account_one_month_max_amount", 0.35)

    if monthly_std > 0:
        deviation_ratio = amount / monthly_std
        add(
            "Account_one_month_std_dev",
            0.60
            if deviation_ratio >= 100
            else 0.40
            if deviation_ratio >= 20
            else 0.20
            if deviation_ratio >= 8
            else 0.0,
        )
    elif amount >= 1_000_000:
        add("Account_one_month_std_dev", 0.35)

    weighted_flags = {
        "Customer_rooting_jailbreak_indicator": 0.35,
        "Customer_mobile_roaming_indicator": 0.18,
        "Customer_VPN_Indicator": 0.50,
        "Customer_inquery_atm_limit": 0.15,
        "Customer_increase_atm_limit": 0.15,
        "Account_indicator_release_limit_excess": 0.25,
        "Account_indicator_Openbanking": 0.15,
        "Account_release_suspention": 0.55,
        "Unused_terminal_status": 0.35,
        "Flag_deposit_more_than_tenMillion": 0.50,
        "Unused_account_status": 0.25,
        "Recipient_account_suspend_status": 0.85,
        "First_time_iOS_by_vulnerable_user": 0.30,
    }
    for feature_name, weight in weighted_flags.items():
        if _is_enabled(features[feature_name]):
            add(feature_name, weight)

    for feature_name in (
        "Customer_flag_change_of_authentication_1",
        "Customer_flag_change_of_authentication_2",
        "Customer_flag_change_of_authentication_3",
        "Customer_flag_change_of_authentication_4",
    ):
        if _is_enabled(features[feature_name]):
            add(feature_name, 0.20)

    for feature_name in (
        "Customer_flag_terminal_malicious_behavior_1",
        "Customer_flag_terminal_malicious_behavior_2",
        "Customer_flag_terminal_malicious_behavior_3",
        "Customer_flag_terminal_malicious_behavior_5",
        "Customer_flag_terminal_malicious_behavior_6",
    ):
        if _is_enabled(features[feature_name]):
            add(feature_name, 0.22)

    connection_failures = max(
        0,
        int(_as_float(features["Transaction_num_connection_failure"])),
    )
    add(
        "Transaction_num_connection_failure",
        min(connection_failures, 5) * 0.12,
    )

    if _is_enabled(features["Another_Person_Account"]):
        add("Another_Person_Account", 0.20)
    transaction_count = max(
        0,
        int(_as_float(features["Number_of_transaction_with_the_account"])),
    )
    transaction_history = max(
        0,
        int(_as_float(features["Transaction_history_with_the_account"])),
    )
    if transaction_count == 0:
        add("Number_of_transaction_with_the_account", 0.15)
    elif transaction_count >= 10:
        add("Number_of_transaction_with_the_account", -0.15)
    if transaction_history == 0:
        add("Transaction_history_with_the_account", 0.15)
    elif transaction_history >= 5:
        add("Transaction_history_with_the_account", -0.20)

    if _is_enabled(features["transaction_is_dawn"]):
        add("transaction_is_dawn", 0.20)

    distance = max(0.0, _as_float(features["Distance"]))
    elapsed_seconds = max(
        0.0,
        _as_float(features["seconds_since_prev_transaction"]),
    )
    if distance >= 50 and elapsed_seconds == 0:
        add("Distance", 0.80)
    elif distance > 0 and elapsed_seconds > 0:
        speed_kmh = distance / (elapsed_seconds / 3600)
        add(
            "Distance",
            1.20
            if speed_kmh >= 500
            else 0.80
            if speed_kmh >= 200
            else 0.35
            if speed_kmh >= 100
            else 0.0,
        )

    if _is_enabled(features["has_resumed_history"]):
        days_since_resumed = _as_float(features["days_since_transaction_resumed"])
        if 0 <= days_since_resumed <= 7:
            add("days_since_transaction_resumed", 0.30)

    return dict(sorted(contributions.items()))


def _stub_shap_contributions(
    features: object,
    risk_contributions: dict[str, float],
) -> dict[str, float]:
    """위험 점수를 합계가 보존되는 91개 Feature 더미 SHAP으로 가공한다.

    고정 가중치가 그대로 노출되지 않도록 Feature의 실제 값으로 결정적인 미세 변동을
    주되, 양수·음수 기여도의 합은 각각 유지한다. 따라서 같은 거래는 항상 같은 값을
    반환하고 ``BASE_LOG_ODDS + sum(shap)``도 확률 계산의 log-odds와 일치한다.
    """

    shap = {str(feature_name): 0.0 for feature_name in features.index}
    for is_positive in (True, False):
        group = {
            feature_name: value
            for feature_name, value in risk_contributions.items()
            if (value > 0) is is_positive
        }
        if not group:
            continue

        varied = {
            feature_name: value
            * _deterministic_shap_scale(feature_name, features[feature_name])
            for feature_name, value in group.items()
        }
        target_sum = math.fsum(group.values())
        varied_sum = math.fsum(varied.values())
        normalization = target_sum / varied_sum
        rounded = {
            feature_name: round(value * normalization, 4)
            for feature_name, value in varied.items()
        }

        # 네 자리 반올림으로 생긴 작은 합계 오차는 가장 큰 기여도에 되돌린다.
        rounding_delta = round(target_sum - math.fsum(rounded.values()), 4)
        anchor = max(group, key=lambda feature_name: abs(group[feature_name]))
        rounded[anchor] = round(rounded[anchor] + rounding_delta, 4)
        shap.update(rounded)

    # 실제 Tree SHAP처럼 직접 위험 규칙에 선택되지 않은 Feature도 작은 양·음수
    # 기여도를 갖게 한다. 배경 기여도의 총합은 0으로 맞춰 예측 확률은 바꾸지 않는다.
    inactive_features = [
        str(feature_name)
        for feature_name in features.index
        if str(feature_name) not in risk_contributions
    ]
    if inactive_features:
        raw_background = {
            feature_name: _deterministic_background_shap(
                feature_name,
                features[feature_name],
            )
            for feature_name in inactive_features
        }
        background_mean = math.fsum(raw_background.values()) / len(raw_background)
        background = {
            feature_name: round(value - background_mean, 4)
            for feature_name, value in raw_background.items()
        }
        rounding_delta = round(-math.fsum(background.values()), 4)
        anchor = max(background, key=lambda feature_name: abs(background[feature_name]))
        background[anchor] = round(background[anchor] + rounding_delta, 4)
        shap.update(background)

    return shap


def _deterministic_shap_scale(feature_name: str, feature_value: object) -> float:
    """Feature 이름과 값만으로 0.85~1.15 범위의 재현 가능한 배율을 만든다."""

    payload = f"{feature_name}={feature_value}".encode()
    digest = hashlib.sha256(payload).digest()
    unit_interval = int.from_bytes(digest[:2], "big") / 65_535
    return 0.85 + (0.30 * unit_interval)


def _deterministic_background_shap(
    feature_name: str,
    feature_value: object,
) -> float:
    """비활성 Feature용 재현 가능한 작은 양·음수 설명값을 만든다."""

    payload = f"background:{feature_name}={feature_value}".encode()
    digest = hashlib.sha256(payload).digest()
    sign = 1.0 if digest[0] >= 128 else -1.0
    magnitude = 0.002 + (0.018 * (digest[1] / 255))
    return sign * magnitude


def _sigmoid(value: float) -> float:
    """log-odds를 0~1 확률로 안정적으로 변환한다."""

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _as_float(value: object) -> float:
    """CSV에서 읽은 숫자 또는 JSON 숫자를 Stub 계산용 실수로 바꾼다."""

    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_enabled(value: object) -> bool:
    """공개 데이터의 0/1 상태값을 안전하게 판별한다."""

    return _as_float(value) == 1.0


def predictor_from_environment() -> Predictor:
    """환경별로 Stub 또는 MLflow 실제 모델을 선택한다."""

    mode = os.getenv("ML_PREDICTOR_MODE", "stub").strip().lower()
    if mode == "stub":
        return StubPredictor.from_environment()
    if mode == "mlflow":
        from fdshield_ml.serving.mlflow_predictor import MLflowPredictor

        return MLflowPredictor.from_environment()
    raise ValueError("ML_PREDICTOR_MODE must be 'stub' or 'mlflow'")
