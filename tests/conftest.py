"""ML 담당자 raw51/model79 계약에서 공유하는 원본 거래 Fixture."""

from collections.abc import Callable

import pytest


def training_row_from_raw51(
    features: dict[str, object],
    *,
    transaction_id: object,
    is_fraud: object,
) -> dict[str, object]:
    """실시간 raw51에 학습 CSV 전용 열을 더해 train1 raw64 한 행을 만든다."""

    row = {
        "transaction_id": transaction_id,
        **features,
        "customer_name": "synthetic-customer",
        "customer_identification_number": f"synthetic-{transaction_id}",
        "account_account_number": f"source-{transaction_id}",
        "error_code": None,
        "ip_address": None,
        "mac_address": None,
        "location": None,
        "recipient_account_number": f"recipient-{transaction_id}",
        "first_time_ios_by_vulnerable_user": 0,
        "customer_id": transaction_id,
        "balance_drain_ratio": 0.1,
        "is_fraud": is_fraud,
    }
    row["account_release_suspention"] = row.pop("recipient_release_suspension")
    row["transaction_resumed_date"] = row.pop(
        "recipient_transaction_resumed_date"
    )
    row["flag_deposit_more_than_tenmillion"] = row.pop(
        "flag_deposit_more_than_ten_million"
    )
    return row


@pytest.fixture
def raw_features_factory() -> Callable[..., dict[str, object]]:
    """거래 ID를 제외한 flat 추론 DTO의 51개 Feature를 만든다."""

    def factory(**overrides: object) -> dict[str, object]:
        features: dict[str, object] = {
            "customer_birth_date": "1981-01-20T00:00:00+09:00",
            "customer_gender": "male",
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
            "account_account_type": "a",
            "account_creation_datetime": "2024-12-02T22:14:22+09:00",
            "account_initial_balance": 8_812_467,
            "account_balance": 4_817_417,
            "account_indicator_release_limit_excess": 1,
            "account_amount_daily_limit": 10_000_000,
            "account_indicator_openbanking": 0,
            "account_remaining_amount_daily_limit_exceeded": 6_004_950,
            "recipient_release_suspension": 0,
            "account_one_month_max_amount": 1_000_000,
            "account_one_month_std_dev": 100_000.0,
            "account_dawn_one_month_max_amount": 500_000,
            "account_dawn_one_month_std_dev": 50_000.0,
            "transaction_datetime": "2025-01-12T03:04:05+09:00",
            "transaction_amount": 100_000,
            "channel": "mobile",
            "operating_system": "android",
            "type_general_automatic": "general",
            "access_medium": "e",
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
            "recipient_transaction_resumed_date": "2025-01-01T03:04:05+09:00",
        }
        features.update(overrides)
        return features

    return factory
