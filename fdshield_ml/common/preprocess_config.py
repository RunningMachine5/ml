"""ML 담당자 전달본의 raw64/raw60 -> model80 전처리 설정.

학습 CSV와 실시간 추론은 식별·메타데이터 유무만 다르고 동일한 80개
모델 Feature를 만든다. 이 모듈의 tuple 순서는 저장된 XGBoost 모델의
``feature_names`` 순서와 같아야 한다.
"""

from __future__ import annotations

TRANSACTION_ID_COLUMN = "transaction_id"
LABEL_COLUMN = "is_fraud"

# ``PredictInputDTO`` 60개 중 응답 상관관계용 transaction_id를 제외한 59개다.
MODEL_INPUT_COLUMNS = (
    "customer_birth_date",
    "customer_gender",
    "customer_name",
    "customer_registration_datetime",
    "customer_credit_rating",
    "customer_flag_change_of_authentication_1",
    "customer_flag_change_of_authentication_2",
    "customer_flag_change_of_authentication_3",
    "customer_flag_change_of_authentication_4",
    "customer_rooting_jailbreak_indicator",
    "customer_mobile_roaming_indicator",
    "customer_vpn_indicator",
    "customer_loan_type",
    "customer_flag_terminal_malicious_behavior_1",
    "customer_flag_terminal_malicious_behavior_2",
    "customer_flag_terminal_malicious_behavior_3",
    "customer_flag_terminal_malicious_behavior_5",
    "customer_flag_terminal_malicious_behavior_6",
    "customer_inquery_atm_limit",
    "customer_increase_atm_limit",
    "account_account_number",
    "account_account_type",
    "account_creation_datetime",
    "account_initial_balance",
    "account_balance",
    "account_indicator_release_limit_excess",
    "account_amount_daily_limit",
    "account_indicator_openbanking",
    "account_remaining_amount_daily_limit_exceeded",
    "account_release_suspention",
    "account_one_month_max_amount",
    "account_one_month_std_dev",
    "account_dawn_one_month_max_amount",
    "account_dawn_one_month_std_dev",
    "transaction_datetime",
    "transaction_amount",
    "channel",
    "operating_system",
    "error_code",
    "type_general_automatic",
    "ip_address",
    "mac_address",
    "access_medium",
    "location",
    "recipient_account_number",
    "transaction_num_connection_failure",
    "another_person_account",
    "distance",
    "time_difference",
    "unused_terminal_status",
    "last_atm_transaction_datetime",
    "last_bank_branch_transaction_datetime",
    "flag_deposit_more_than_ten_million",
    "unused_account_status",
    "recipient_account_suspend_status",
    "number_of_transaction_with_the_account",
    "transaction_history_with_the_account",
    "first_time_ios_by_vulnerable_user",
    "transaction_resumed_date",
)
MODEL_INPUT_COLUMN_SET = frozenset(MODEL_INPUT_COLUMNS)

# ML 담당자의 flat PredictInputDTO 전체 계약.
SERVING_INPUT_COLUMNS = (TRANSACTION_ID_COLUMN, *MODEL_INPUT_COLUMNS)
SERVING_INPUT_COLUMN_SET = frozenset(SERVING_INPUT_COLUMNS)

# train1.csv에만 추가되는 모델 비입력 컬럼이다.
TRAINING_METADATA_COLUMNS = (
    "customer_identification_number",
    "customer_id",
    "balance_drain_ratio",
    LABEL_COLUMN,
)

# train1.csv의 의미상 올바른 canonical 64열 순서다.
TRAINING_INPUT_COLUMNS = (
    TRANSACTION_ID_COLUMN,
    "customer_birth_date",
    "customer_gender",
    "customer_name",
    "customer_identification_number",
    *MODEL_INPUT_COLUMNS[3:],
    "customer_id",
    "balance_drain_ratio",
    LABEL_COLUMN,
)

# 전달된 원본 파일에는 이 한 컬럼만 오타가 있다. 원본을 변형하지 않고
# 로딩 직후 canonical 이름으로 바꾼다.
CSV_ALIAS_COLUMNS = {
    "flag_deposit_more_than_tenmillion": "flag_deposit_more_than_ten_million",
}
RAW_TRAINING_INPUT_COLUMNS = tuple(
    next(
        (
            alias
            for alias, canonical in CSV_ALIAS_COLUMNS.items()
            if canonical == column
        ),
        column,
    )
    for column in TRAINING_INPUT_COLUMNS
)

CATEGORICAL_LEVELS = {
    "customer_gender": ("male", "female"),
    "customer_loan_type": ("a", "b", "c", "d", "e"),
    "account_account_type": ("a", "b", "c", "d"),
    "channel": ("mobile", "internet", "atm", "others"),
    "operating_system": (
        "android",
        "ios",
        "windows",
        "macos",
        "linux",
        "others",
    ),
    "type_general_automatic": ("general", "automatic"),
    "access_medium": ("a", "b", "c", "d", "e", "f", "g", "h"),
}

TRANSACTION_DATETIME_COLUMN = "transaction_datetime"
REQUIRED_ELAPSED_COLUMNS = {
    "customer_registration_datetime": "days_since_registration",
    "account_creation_datetime": "days_since_account_creation",
}
OPTIONAL_ELAPSED_COLUMNS = {
    "last_atm_transaction_datetime": "days_since_last_atm",
    "last_bank_branch_transaction_datetime": "days_since_last_bank_branch",
    "transaction_resumed_date": "days_since_transaction_resumed",
}

# 원본 이름 그대로 80개 벡터에 들어가는 숫자 필드다. transaction_amount는
# 금액 파생 Feature와 한 번에 생성하므로 여기서는 제외한다.
NUMERIC_PASSTHROUGH_COLUMNS = (
    "customer_credit_rating",
    "customer_flag_change_of_authentication_1",
    "customer_flag_change_of_authentication_2",
    "customer_flag_change_of_authentication_3",
    "customer_flag_change_of_authentication_4",
    "customer_rooting_jailbreak_indicator",
    "customer_mobile_roaming_indicator",
    "customer_vpn_indicator",
    "customer_flag_terminal_malicious_behavior_1",
    "customer_flag_terminal_malicious_behavior_2",
    "customer_flag_terminal_malicious_behavior_3",
    "customer_flag_terminal_malicious_behavior_5",
    "customer_flag_terminal_malicious_behavior_6",
    "customer_inquery_atm_limit",
    "customer_increase_atm_limit",
    "account_indicator_release_limit_excess",
    "account_indicator_openbanking",
    "account_remaining_amount_daily_limit_exceeded",
    "account_release_suspention",
    "transaction_num_connection_failure",
    "another_person_account",
    "unused_terminal_status",
    "flag_deposit_more_than_ten_million",
    "unused_account_status",
    "recipient_account_suspend_status",
    "number_of_transaction_with_the_account",
    "transaction_history_with_the_account",
    "first_time_ios_by_vulnerable_user",
)

# xgb_model_weights_12_24_24.json의 feature_names와 동일한 80개 순서다.
MODEL_FEATURE_COLUMNS = (
    "customer_age",
    "customer_credit_rating",
    "customer_flag_change_of_authentication_1",
    "customer_flag_change_of_authentication_2",
    "customer_flag_change_of_authentication_3",
    "customer_flag_change_of_authentication_4",
    "customer_rooting_jailbreak_indicator",
    "customer_mobile_roaming_indicator",
    "customer_vpn_indicator",
    "customer_flag_terminal_malicious_behavior_1",
    "customer_flag_terminal_malicious_behavior_2",
    "customer_flag_terminal_malicious_behavior_3",
    "customer_flag_terminal_malicious_behavior_5",
    "customer_flag_terminal_malicious_behavior_6",
    "customer_inquery_atm_limit",
    "customer_increase_atm_limit",
    "account_indicator_release_limit_excess",
    "account_indicator_openbanking",
    "account_remaining_amount_daily_limit_exceeded",
    "account_release_suspention",
    "transaction_num_connection_failure",
    "another_person_account",
    "seconds_since_last_transaction",
    "distance_since_last_transaction",
    "distance_per_minute",
    "unused_terminal_status",
    "flag_deposit_more_than_ten_million",
    "unused_account_status",
    "recipient_account_suspend_status",
    "number_of_transaction_with_the_account",
    "transaction_history_with_the_account",
    "first_time_ios_by_vulnerable_user",
    "transaction_hour",
    "transaction_day",
    "transaction_day_of_week",
    "transaction_is_dawn",
    "transaction_is_weekend",
    "days_since_registration",
    "days_since_account_creation",
    "days_since_last_atm",
    "days_since_last_bank_branch",
    "days_since_transaction_resumed",
    "transaction_amount",
    "amount_to_balance_ratio",
    "amount_to_daily_limit_ratio",
    "amount_to_one_month_max_ratio",
    "amount_to_one_month_std_dev_ratio",
    "amount_to_dawn_one_month_max_ratio",
    "amount_to_dawn_one_month_std_dev_ratio",
    "customer_gender_male",
    "customer_gender_female",
    "customer_loan_type_a",
    "customer_loan_type_b",
    "customer_loan_type_c",
    "customer_loan_type_d",
    "customer_loan_type_e",
    "account_account_type_a",
    "account_account_type_b",
    "account_account_type_c",
    "account_account_type_d",
    "channel_mobile",
    "channel_internet",
    "channel_atm",
    "channel_others",
    "operating_system_android",
    "operating_system_ios",
    "operating_system_windows",
    "operating_system_macos",
    "operating_system_linux",
    "operating_system_others",
    "type_general_automatic_general",
    "type_general_automatic_automatic",
    "access_medium_a",
    "access_medium_b",
    "access_medium_c",
    "access_medium_d",
    "access_medium_e",
    "access_medium_f",
    "access_medium_g",
    "access_medium_h",
)

# 추론 features 안에는 학습 전용 메타데이터와 라벨을 허용하지 않는다.
FORBIDDEN_INFERENCE_COLUMNS = frozenset(
    {
        "customer_identification_number",
        "customer_id",
        "balance_drain_ratio",
        LABEL_COLUMN,
        *CSV_ALIAS_COLUMNS,
    }
)


if len(MODEL_INPUT_COLUMNS) != 59:  # pragma: no cover - import invariant
    raise RuntimeError("MODEL_INPUT_COLUMNS must contain exactly 59 columns.")
if len(SERVING_INPUT_COLUMNS) != 60:  # pragma: no cover - import invariant
    raise RuntimeError("SERVING_INPUT_COLUMNS must contain exactly 60 columns.")
if len(TRAINING_INPUT_COLUMNS) != 64:  # pragma: no cover - import invariant
    raise RuntimeError("TRAINING_INPUT_COLUMNS must contain exactly 64 columns.")
if len(MODEL_FEATURE_COLUMNS) != 80:  # pragma: no cover - import invariant
    raise RuntimeError("MODEL_FEATURE_COLUMNS must contain exactly 80 columns.")
if len(MODEL_FEATURE_COLUMNS) != len(set(MODEL_FEATURE_COLUMNS)):
    raise RuntimeError("MODEL_FEATURE_COLUMNS must not contain duplicates.")


__all__ = [
    "CATEGORICAL_LEVELS",
    "CSV_ALIAS_COLUMNS",
    "FORBIDDEN_INFERENCE_COLUMNS",
    "LABEL_COLUMN",
    "MODEL_FEATURE_COLUMNS",
    "MODEL_INPUT_COLUMNS",
    "MODEL_INPUT_COLUMN_SET",
    "NUMERIC_PASSTHROUGH_COLUMNS",
    "OPTIONAL_ELAPSED_COLUMNS",
    "RAW_TRAINING_INPUT_COLUMNS",
    "REQUIRED_ELAPSED_COLUMNS",
    "SERVING_INPUT_COLUMNS",
    "SERVING_INPUT_COLUMN_SET",
    "TRAINING_INPUT_COLUMNS",
    "TRAINING_METADATA_COLUMNS",
    "TRANSACTION_DATETIME_COLUMN",
    "TRANSACTION_ID_COLUMN",
]
