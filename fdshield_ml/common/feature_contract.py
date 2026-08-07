"""Backend와 ML이 공유하는 실시간 추론 입력·모델 Feature 계약."""

from __future__ import annotations


# Backend가 검증한 뒤 ML Serving으로 전달하는 전처리 전 원본 Feature 54개다.
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
    "Type_General_Automatic",
    "Access_Medium",
    "Location",
    "Transaction_num_connection_failure",
    "Another_Person_Account",
    "Distance",
    "Time Difference",
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

CATEGORICAL_LEVELS = {
    "Customer_Gender": ("male", "female"),
    "Customer_loan_type": ("a", "b", "c", "d", "e"),
    "Account_account_type": ("a", "b", "c", "d"),
    "Channel": ("mobile", "internet", "ATM", "Others"),
    "Operating_System": ("Android", "iOS", "Windows", "macOS", "Linux", "Others"),
    "Error_Code": ("a", "b", "c", "d", "e", "f"),
    "Type_General_Automatic": ("general", "automatic"),
    "Access_Medium": ("a", "b", "c", "d", "e", "f", "g", "h"),
}

SOURCE_DATETIME_COLUMNS = (
    "Transaction_Datetime",
    "Customer_registration_datetime",
    "Account_creation_datetime",
    "Last_atm_transaction_datetime",
    "Last_bank_branch_transaction_datetime",
    "Transaction_resumed_date",
)

TRANSACTION_DATETIME_COLUMN = "Transaction_Datetime"
REQUIRED_ELAPSED_COLUMNS = {
    "Customer_registration_datetime": "days_since_registration",
    "Account_creation_datetime": "days_since_account_creation",
}
OPTIONAL_ELAPSED_COLUMNS = {
    "Last_atm_transaction_datetime": ("days_since_last_atm", "has_atm_history"),
    "Last_bank_branch_transaction_datetime": (
        "days_since_last_branch",
        "has_branch_history",
    ),
    "Transaction_resumed_date": (
        "days_since_transaction_resumed",
        "has_resumed_history",
    ),
}

DERIVED_FEATURE_COLUMNS = (
    "transaction_hour",
    "transaction_day",
    "transaction_dayofweek",
    "transaction_is_dawn",
    "transaction_is_weekend",
    "days_since_registration",
    "days_since_account_creation",
    "days_since_last_atm",
    "has_atm_history",
    "days_since_last_branch",
    "has_branch_history",
    "days_since_transaction_resumed",
    "has_resumed_history",
    "seconds_since_prev_transaction",
    "location_latitude",
    "location_longitude",
)

ONE_HOT_FEATURE_COLUMNS = tuple(
    f"{column}_{level}"
    for column, levels in CATEGORICAL_LEVELS.items()
    for level in levels
)

_NON_PASSTHROUGH_COLUMNS = frozenset(
    (*SOURCE_DATETIME_COLUMNS, "Time Difference", "Location", *CATEGORICAL_LEVELS)
)
NUMERIC_PASSTHROUGH_COLUMNS = tuple(
    column for column in MODEL_INPUT_COLUMNS if column not in _NON_PASSTHROUGH_COLUMNS
)

# 실제 모델이 받는 순서가 고정된 숫자 Feature 91개다.
MODEL_FEATURE_COLUMNS = (
    *NUMERIC_PASSTHROUGH_COLUMNS,
    *DERIVED_FEATURE_COLUMNS,
    *ONE_HOT_FEATURE_COLUMNS,
)

# 정답·식별정보와 폐기된 구계약 컬럼은 실시간 요청에서 허용하지 않는다.
FORBIDDEN_INFERENCE_COLUMNS = frozenset(
    {
        "Fraud_Type",
        "Is_Fraud",
        "ID",
        "Customer_ID",
        "Customer_personal_identifier",
        "Customer_identification_number",
        "Account_account_number",
        "IP_Address",
        "MAC_Address",
        "Recipient_Account_Number",
        "Time_difference",
        "Transaction_Failure_Status",
        "Customer_flag_terminal_malicious_behavior_4",
    }
)


if len(MODEL_INPUT_COLUMNS) != 54:  # pragma: no cover - import-time invariant
    raise RuntimeError("MODEL_INPUT_COLUMNS must contain exactly 54 columns.")
if len(MODEL_FEATURE_COLUMNS) != 91:  # pragma: no cover - import-time invariant
    raise RuntimeError("MODEL_FEATURE_COLUMNS must contain exactly 91 columns.")
