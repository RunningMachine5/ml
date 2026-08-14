"""ML 담당자가 정의한 정식 flat raw60 추론 입력 DTO."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FeatureValue = str | int | float | bool | datetime | timedelta | None

# 고객 원장이 아직 연결되지 않은 거래에 사용할 임시 프로필이다.
# 모델이 허용하는 범주 안에서 가장 평범한 값만 선택했다. 고객 API가 완성되면
# 요청에 들어온 실제 값이 우선되므로 이 기본값은 자동으로 사용되지 않는다.
CUSTOMER_FEATURE_DEFAULTS: dict[str, FeatureValue] = {
    "customer_birth_date": datetime(1990, 1, 1, tzinfo=UTC),
    "customer_gender": "male",
    "customer_name": "unknown-customer",
    "customer_registration_datetime": datetime(2020, 1, 1, tzinfo=UTC),
    "customer_credit_rating": 5,
    "customer_loan_type": "a",
}

# Backend가 아직 계산하지 못하는 파생 Feature의 임시 기본값이다.
# 원천 거래정보를 만드는 값이 아니라, 실제 파생 계산기가 완성될 때까지
# ML 추론을 중단하지 않기 위한 중립값만 모아 둔다.
DERIVED_FEATURE_DEFAULTS: dict[str, FeatureValue] = {
    "customer_flag_change_of_authentication_1": False,
    "customer_flag_change_of_authentication_2": False,
    "customer_flag_change_of_authentication_3": False,
    "customer_flag_change_of_authentication_4": False,
    "customer_inquery_atm_limit": False,
    "customer_increase_atm_limit": False,
    "account_release_suspention": False,
    "account_one_month_max_amount": 0.0,
    "account_one_month_std_dev": 0.0,
    "account_dawn_one_month_max_amount": 0.0,
    "account_dawn_one_month_std_dev": 0.0,
    "distance": 0.0,
    "time_difference": timedelta(0),
    "unused_terminal_status": False,
    "last_atm_transaction_datetime": None,
    "last_bank_branch_transaction_datetime": None,
    "flag_deposit_more_than_ten_million": False,
    "unused_account_status": False,
    "recipient_account_suspend_status": False,
    "number_of_transaction_with_the_account": 0,
    "transaction_history_with_the_account": 0,
    "first_time_ios_by_vulnerable_user": False,
    "transaction_resumed_date": None,
}

# ATM 출금처럼 상대 계좌가 없는 거래를 나타내는 임시 식별값이다.
# 실제 계좌번호와 혼동되지 않도록 일반 계좌번호 형식이 아닌 고정 문자열을 쓴다.
RECIPIENT_ACCOUNT_DEFAULT = "unknown-recipient"

# Serving에서만 사용하는 전체 임시값 목록이다. 학습 CSV에는 적용하지 않는다.
SERVING_FEATURE_DEFAULTS = {
    **CUSTOMER_FEATURE_DEFAULTS,
    **DERIVED_FEATURE_DEFAULTS,
    "recipient_account_number": RECIPIENT_ACCOUNT_DEFAULT,
}


class PredictInputDTO(BaseModel):
    """ML 담당자가 정의한 정식 flat snake_case 60개 추론 요청.

    계좌·거래 원천값은 반드시 받아야 한다. 아직 연결되지 않은 고객 프로필과
    계산기가 없는 파생값만 임시 기본값으로 보완하여 실제 모델 추론까지 진행한다.
    """

    model_config = ConfigDict(extra="forbid")

    # Backend DB가 생성한 양의 정수 PK만 추론 상관관계 ID로 허용한다.
    transaction_id: int = Field(strict=True, gt=0)
    customer_birth_date: datetime = datetime(1990, 1, 1, tzinfo=UTC)
    customer_gender: str = "male"
    customer_name: str = "unknown-customer"
    customer_registration_datetime: datetime = datetime(2020, 1, 1, tzinfo=UTC)
    customer_credit_rating: int = 5
    customer_flag_change_of_authentication_1: bool = False
    customer_flag_change_of_authentication_2: bool = False
    customer_flag_change_of_authentication_3: bool = False
    customer_flag_change_of_authentication_4: bool = False
    customer_rooting_jailbreak_indicator: bool
    customer_mobile_roaming_indicator: bool
    customer_vpn_indicator: bool
    customer_loan_type: str = "a"
    customer_flag_terminal_malicious_behavior_1: bool
    customer_flag_terminal_malicious_behavior_2: bool
    customer_flag_terminal_malicious_behavior_3: bool
    customer_flag_terminal_malicious_behavior_5: bool
    customer_flag_terminal_malicious_behavior_6: bool
    customer_inquery_atm_limit: bool = False
    customer_increase_atm_limit: bool = False
    account_account_number: str | int
    account_account_type: str
    account_creation_datetime: datetime
    account_initial_balance: float | None
    account_balance: float | None
    account_indicator_release_limit_excess: int
    account_amount_daily_limit: float
    # 전달 DTO의 bool 표기는 CSV·학습 모델과 달라 금액형으로 바로잡는다.
    account_remaining_amount_daily_limit_exceeded: float | None
    account_indicator_openbanking: bool
    account_release_suspention: bool = False
    account_one_month_max_amount: float = 0.0
    account_one_month_std_dev: float = 0.0
    account_dawn_one_month_max_amount: float = 0.0
    account_dawn_one_month_std_dev: float = 0.0
    transaction_datetime: datetime
    transaction_amount: float
    channel: str
    operating_system: str | None
    error_code: str = Field(max_length=8)
    type_general_automatic: str
    ip_address: str | None
    mac_address: str | None
    access_medium: str | None
    location: str
    recipient_account_number: str | int = RECIPIENT_ACCOUNT_DEFAULT
    transaction_num_connection_failure: int
    another_person_account: bool
    distance: float = 0.0
    time_difference: timedelta = timedelta(0)
    unused_terminal_status: bool = False
    last_atm_transaction_datetime: datetime | None = None
    last_bank_branch_transaction_datetime: datetime | None = None
    flag_deposit_more_than_ten_million: bool = False
    unused_account_status: bool = False
    recipient_account_suspend_status: bool = False
    number_of_transaction_with_the_account: int = 0
    transaction_history_with_the_account: int = 0
    first_time_ios_by_vulnerable_user: bool = False
    transaction_resumed_date: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_missing_serving_features(cls, value: object) -> object:
        """누락·null 고객·파생값을 임시 기본값으로 바꾼다.

        Pydantic의 필드 기본값은 키가 아예 없을 때만 적용된다. JSON에서
        ``null``이 온 경우도 같은 '미계산 상태'이므로 여기서 함께 보완한다.
        계좌번호·거래금액 같은 원천값은 이 목록에 없으므로 계속 필수다.
        """

        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        for field_name, default_value in SERVING_FEATURE_DEFAULTS.items():
            current_value = normalized.get(field_name)
            if current_value is None or (
                isinstance(current_value, str) and not current_value.strip()
            ):
                normalized[field_name] = default_value
        return normalized

    @field_validator(
        "account_initial_balance",
        "account_balance",
        "account_remaining_amount_daily_limit_exceeded",
        "access_medium",
        mode="before",
    )
    @classmethod
    def normalize_optional_train1_value(cls, value: object) -> object | None:
        """CSV 빈칸과 JSON null을 같은 missing 입력으로 취급한다."""

        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    @field_validator(
        "account_remaining_amount_daily_limit_exceeded",
        mode="before",
    )
    @classmethod
    def reject_boolean_remaining_daily_limit(cls, value: object) -> object:
        """잘못된 bool이 금액 0/1로 조용히 변환되는 것을 막는다."""

        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004
                "account_remaining_amount_daily_limit_exceeded must be an amount, "
                "not a boolean"
            )
        return value

    def feature_values(self) -> dict[str, FeatureValue]:
        """식별용 transaction_id를 제외한 공용 전처리 입력을 반환한다."""

        return self.model_dump(exclude={"transaction_id"})


__all__ = [
    "CUSTOMER_FEATURE_DEFAULTS",
    "DERIVED_FEATURE_DEFAULTS",
    "RECIPIENT_ACCOUNT_DEFAULT",
    "SERVING_FEATURE_DEFAULTS",
    "PredictInputDTO",
]
