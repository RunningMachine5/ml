"""현재 공개 데이터 기준 실시간 추론 입력 계약."""

from __future__ import annotations


# test.csv의 원본 컬럼에서 학습 코드가 제외하는 식별정보 8개를 뺀 목록이다.
# 실제 모델을 연결할 때도 이 컬럼 이름과 원본 값을 그대로 Pipeline에 전달한다.
MODEL_INPUT_COLUMNS = (
    "Customer_Birthyear",
    "Customer_Gender",
    "Customer_registration_datetime",
    "Customer_credit_rating",
    "Customer_flag_change_of_authentication_1",
    "Customer_flag_change_of_authentication_2",
    "Customer_flag_change_of_authentication_3",
    "Customer_flag_change_of_authentication_4",
    "Customer_rooting_jailbreak_indicator",
    "Customer_mobile_roaming_indicator",
    "Customer_VPN_Indicator",
    "Customer_loan_type",
    "Customer_flag_terminal_malicious_behavior_1",
    "Customer_flag_terminal_malicious_behavior_2",
    "Customer_flag_terminal_malicious_behavior_3",
    "Customer_flag_terminal_malicious_behavior_4",
    "Customer_flag_terminal_malicious_behavior_5",
    "Customer_flag_terminal_malicious_behavior_6",
    "Customer_inquery_atm_limit",
    "Customer_increase_atm_limit",
    "Account_account_type",
    "Account_creation_datetime",
    "Account_initial_balance",
    "Account_balance",
    "Account_indicator_release_limit_excess",
    "Account_amount_daily_limit",
    "Account_indicator_Openbanking",
    "Account_remaining_amount_daily_limit_exceeded",
    "Account_release_suspention",
    "Account_one_month_max_amount",
    "Account_one_month_std_dev",
    "Account_dawn_one_month_max_amount",
    "Account_dawn_one_month_std_dev",
    "Transaction_Datetime",
    "Transaction_Amount",
    "Channel",
    "Operating_System",
    "Error_Code",
    "Transaction_Failure_Status",
    "Type_General_Automatic",
    "Access_Medium",
    "Transaction_num_connection_failure",
    "Another_Person_Account",
    "Distance",
    "Time_difference",
    "Unused_terminal_status",
    "Last_atm_transaction_datetime",
    "Last_bank_branch_transaction_datetime",
    "Flag_deposit_more_than_tenMillion",
    "Unused_account_status",
    "Recipient_account_suspend_status",
    "Number_of_transaction_with_the_account",
    "Transaction_history_with_the_account",
    "First_time_iOS_by_vulnerable_user",
    "Transaction_resumed_date",
)

MODEL_INPUT_COLUMN_SET = frozenset(MODEL_INPUT_COLUMNS)

# 정답 라벨과 현재 모델이 사용하지 않는 식별정보는 추론 서버로 보내지 않는다.
FORBIDDEN_INFERENCE_COLUMNS = frozenset(
    {
        "Fraud_Type",
        "ID",
        "Customer_personal_identifier",
        "Customer_identification_number",
        "Account_account_number",
        "IP_Address",
        "MAC_Address",
        "Location",
        "Recipient_Account_Number",
    }
)
