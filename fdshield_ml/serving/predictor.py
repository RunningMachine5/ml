"""실제 모델 연결 전 API 흐름을 검증하는 예측기 스켈레톤."""

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
    """실제 모델 연결 전 전처리 결과의 일부 위험 신호만 사용하는 결정적 Stub."""

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
        """로컬과 Cloud Run에서 동일한 환경변수로 Stub 설정을 생성한다."""

        return cls(
            threshold=float(os.getenv("ML_FRAUD_THRESHOLD", "0.55")),
            model_name=os.getenv(
                "ML_MODEL_NAME",
                "fdshield-rule-based-stub",
            ),
            model_version=os.getenv("ML_MODEL_VERSION", "").strip() or "0",
        )

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """54개 원본 값을 91개로 전처리한 뒤 임시 규칙 점수를 계산한다."""

        # 실제 모델이 연결되면 이 DataFrame을 그대로 predict_proba에 전달한다.
        model_features = preprocess_transaction_features(request.features)
        features = model_features.iloc[0]
        amount = abs(_as_float(features["Transaction_Amount"]))
        standard_deviation = max(
            abs(_as_float(features["Account_one_month_std_dev"])),
            1.0,
        )
        deviation_ratio = amount / standard_deviation

        # 아래 값은 실제 SHAP가 아니라 API 연결을 확인하기 위한 임시 기여도다.
        contributions: dict[str, float] = {}
        contributions["Transaction_Amount"] = (
            0.35 if amount >= 10_000_000 else 0.20 if amount >= 1_000_000 else 0.05
        )
        contributions["Account_one_month_std_dev"] = (
            0.25 if deviation_ratio >= 100 else 0.10 if deviation_ratio >= 10 else 0.0
        )

        indicator_weights = {
            "Customer_VPN_Indicator": 0.15,
            "Unused_terminal_status": 0.15,
            "Recipient_account_suspend_status": 0.10,
        }
        for feature_name, weight in indicator_weights.items():
            contributions[feature_name] = (
                weight if _is_enabled(features[feature_name]) else 0.0
            )

        probability = round(min(0.05 + sum(contributions.values()), 0.99), 2)

        return PredictionResponse(
            transaction_id=request.transaction_id,
            is_fraud=probability >= self.threshold,
            fraud_probability=probability,
            shap=contributions,
            model_name=self.model_name,
            model_version=self.model_version,
        )


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
