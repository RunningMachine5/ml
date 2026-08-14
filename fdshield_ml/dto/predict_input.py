"""ML 담당자가 정의한 정식 flat raw60 추론 입력 DTO."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator

FeatureValue = str | int | float | bool | datetime | timedelta | None


class PredictInputDTO(BaseModel):
    """ML 담당자가 정의한 정식 flat snake_case 60개 추론 요청."""

    model_config = ConfigDict(extra="forbid")

    # Backend DB가 생성한 양의 정수 PK만 추론 상관관계 ID로 허용한다.
    transaction_id: int = Field(strict=True, gt=0)
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
    account_initial_balance: float | None
    account_balance: float | None
    account_indicator_release_limit_excess: int
    account_amount_daily_limit: float
    # 전달 DTO의 bool 표기는 CSV·학습 모델과 달라 금액형으로 바로잡는다.
    account_remaining_amount_daily_limit_exceeded: float | None
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
    error_code: str = Field(max_length=8)
    type_general_automatic: str
    ip_address: str | None
    mac_address: str | None
    access_medium: str | None
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


__all__ = ["PredictInputDTO"]
