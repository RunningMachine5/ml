"""ML 담당자 raw60/model80 계약에서 공유하는 원본 거래 Fixture."""

from collections.abc import Callable

import pytest


@pytest.fixture
def raw_features_factory() -> Callable[..., dict[str, object]]:
    """거래 ID를 제외한 flat 추론 DTO의 59개 Feature를 만든다."""

    def factory(**overrides: object) -> dict[str, object]:
        features: dict[str, object] = {
            "customer_birth_date": "1981-01-20T00:00:00+09:00",
            "customer_gender": "male",
            "customer_name": "test-customer",
            "customer_registration_datetime": "2023-01-20T09:41:55+09:00",
            "customer_credit_rating": 6,
            "customer_flag_change_of_authentication_1": 0,
            "customer_flag_change_of_authentication_2": 0,
            "customer_flag_change_of_authentication_3": 0,
            "customer_flag_change_of_authentication_4": 0,
            "customer_rooting_jailbreak_indicator": 1,
            "customer_mobile_roaming_indicator": 0,
            "customer_vpn_indicator": 0,
            "customer_loan_type": "c",
            "customer_flag_terminal_malicious_behavior_1": 0,
            "customer_flag_terminal_malicious_behavior_2": 0,
            "customer_flag_terminal_malicious_behavior_3": 0,
            "customer_flag_terminal_malicious_behavior_5": 0,
            "customer_flag_terminal_malicious_behavior_6": 0,
            "customer_inquery_atm_limit": 0,
            "customer_increase_atm_limit": 1,
            "account_account_number": "account-1",
            "account_account_type": "a",
            "account_creation_datetime": "2024-12-02T22:14:22+09:00",
            "account_initial_balance": 8_812_467,
            "account_balance": 4_817_417,
            "account_indicator_release_limit_excess": 1,
            "account_amount_daily_limit": 10_000_000,
            "account_indicator_openbanking": 0,
            "account_remaining_amount_daily_limit_exceeded": 6_004_950,
            "account_release_suspention": 0,
            "account_one_month_max_amount": 1_000_000,
            "account_one_month_std_dev": 100_000.0,
            "account_dawn_one_month_max_amount": 500_000,
            "account_dawn_one_month_std_dev": 50_000.0,
            "transaction_datetime": "2025-01-12T03:04:05+09:00",
            "transaction_amount": 100_000,
            "channel": "mobile",
            "operating_system": "Android",
            "error_code": "none",
            "type_general_automatic": "general",
            "ip_address": "127.0.0.1",
            "mac_address": "00:00:00:00:00:00",
            "access_medium": "e",
            "location": "seoul",
            "recipient_account_number": "recipient-1",
            "transaction_num_connection_failure": 0,
            "another_person_account": 1,
            "distance": 0.0,
            "time_difference": "1 days 02:03:04",
            "unused_terminal_status": 0,
            "last_atm_transaction_datetime": "2025-01-10T03:04:05+09:00",
            "last_bank_branch_transaction_datetime": None,
            "flag_deposit_more_than_ten_million": 0,
            "unused_account_status": 1,
            "recipient_account_suspend_status": 0,
            "number_of_transaction_with_the_account": 0,
            "transaction_history_with_the_account": 0,
            "first_time_ios_by_vulnerable_user": 0,
            "transaction_resumed_date": "2025-01-01T03:04:05+09:00",
        }
        features.update(overrides)
        return features

    return factory
