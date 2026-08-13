"""ML 담당자가 정의한 정식 flat raw60 추론 입력 DTO."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, field_validator

FeatureValue = str | int | float | bool | datetime | timedelta | None


def _normalize_transaction_id(value: object) -> str:
    """숫자·문자 거래 ID를 손실 없는 내부 문자열 ID로 통일한다."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        # Pydantic v2는 validator의 TypeError를 ValidationError로 감싸지 않는다.
        raise ValueError(  # noqa: TRY004
            "transaction_id must be a non-empty string or integer"
        )
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("transaction_id must not be empty")
    return normalized


class PredictInputDTO(BaseModel):
    """ML 담당자가 정의한 정식 flat snake_case 60개 추론 요청."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    customer_birth_date: datetime
    customer_gender: str
    customer_name: str
    customer_registration_datetime: datetime
    customer_credit_rating: int
    customer_flag_change_of_authentication_1: bool
    customer_flag_change_of_authentication_2: bool
    customer_flag_change_of_authentication_3: bool
    customer_flag_change_of_authentication_4: bool
    customer_rooting_jailbreak_indicator: bool
    customer_mobile_roaming_indicator: bool
    customer_vpn_indicator: bool
    customer_loan_type: str
    customer_flag_terminal_malicious_behavior_1: bool
    customer_flag_terminal_malicious_behavior_2: bool
    customer_flag_terminal_malicious_behavior_3: bool
    customer_flag_terminal_malicious_behavior_5: bool
    customer_flag_terminal_malicious_behavior_6: bool
    customer_inquery_atm_limit: bool
    customer_increase_atm_limit: bool
    account_account_number: str | int
    account_account_type: str
    account_creation_datetime: datetime
    account_initial_balance: float
    account_balance: float
    account_indicator_release_limit_excess: int
    account_amount_daily_limit: float
    # 전달 DTO의 bool 표기는 CSV·학습 모델과 달라 금액형으로 바로잡는다.
    account_remaining_amount_daily_limit_exceeded: float
    account_indicator_openbanking: bool
    account_release_suspention: bool
    account_one_month_max_amount: float
    account_one_month_std_dev: float
    account_dawn_one_month_max_amount: float
    account_dawn_one_month_std_dev: float
    transaction_datetime: datetime
    transaction_amount: float
    channel: str
    operating_system: str | None
    error_code: str
    type_general_automatic: str
    ip_address: str | None
    mac_address: str | None
    access_medium: str
    location: str
    recipient_account_number: str | int
    transaction_num_connection_failure: int
    another_person_account: bool
    distance: float
    time_difference: timedelta
    unused_terminal_status: bool
    last_atm_transaction_datetime: datetime | None
    last_bank_branch_transaction_datetime: datetime | None
    flag_deposit_more_than_ten_million: bool
    unused_account_status: bool
    recipient_account_suspend_status: bool
    number_of_transaction_with_the_account: int
    transaction_history_with_the_account: int
    first_time_ios_by_vulnerable_user: bool
    transaction_resumed_date: datetime | None

    @field_validator("transaction_id", mode="before")
    @classmethod
    def normalize_transaction_id(cls, value: object) -> str:
        return _normalize_transaction_id(value)

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


__all__ = ["PredictInputDTO"]
