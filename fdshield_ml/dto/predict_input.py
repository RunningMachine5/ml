"""Backend 담당자가 계산한 정식 flat raw51 추론 입력 DTO."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

FeatureValue = str | int | float | bool | datetime | timedelta | None

class PredictInputDTO(BaseModel):
    """거래 ID와 Backend raw51을 받는 추론 요청.

    모델 전처리에 쓰지 않는 고객명·계좌번호·접속 식별값은 받지 않는다.
    """

    model_config = ConfigDict(extra="forbid")

    # Backend DB가 생성한 양의 정수 PK만 추론 상관관계 ID로 허용한다.
    transaction_id: int = Field(strict=True, gt=0)
    customer_birth_date: datetime
    customer_gender: str
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
    account_account_type: str
    account_creation_datetime: datetime
    account_initial_balance: int
    account_balance: int
    account_indicator_release_limit_excess: int
    account_amount_daily_limit: int
    account_remaining_amount_daily_limit_exceeded: int
    account_indicator_openbanking: bool
    recipient_release_suspension: bool
    account_one_month_max_amount: int
    account_one_month_std_dev: float
    account_dawn_one_month_max_amount: int
    account_dawn_one_month_std_dev: float
    transaction_datetime: datetime
    transaction_amount: int
    channel: str
    operating_system: str | None
    type_general_automatic: str
    access_medium: str | None
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
    recipient_transaction_resumed_date: datetime | None

    def feature_values(self) -> dict[str, FeatureValue]:
        """식별용 transaction_id를 제외한 공용 전처리 입력을 반환한다."""

        return self.model_dump(exclude={"transaction_id"})
