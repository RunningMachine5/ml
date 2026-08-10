"""모델 서빙 스켈레톤의 HTTP 계약 테스트."""

import math
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from fdshield_ml.common.feature_contract import (
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
)
from fdshield_ml.serving.app import create_app
from fdshield_ml.serving.predictor import StubPredictor

RawFeaturesFactory = Callable[..., dict[str, object]]


def test_health_and_readiness() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_predict_returns_deterministic_stub_contract(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_000001",
            "features": raw_features_factory(
                Transaction_Amount=50_000_000,
                Account_one_month_std_dev=100_000,
                Customer_VPN_Indicator=1,
                Unused_terminal_status=1,
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "TEST_000001"
    assert body["is_fraud"] is True
    assert body["fraud_probability"] > 0.85
    assert body["model_name"] == "fdshield-rule-based-stub"
    assert body["model_version"] == "0"
    assert body["shap"]["Transaction_Amount"] > 0
    assert body["shap"]["Account_amount_daily_limit"] > 0
    assert body["shap"]["Account_one_month_std_dev"] > 0
    assert body["shap"]["Customer_VPN_Indicator"] > 0
    assert body["shap"]["Unused_terminal_status"] > 0
    assert set(body["shap"]) == set(MODEL_FEATURE_COLUMNS)
    assert sum(value != 0.0 for value in body["shap"].values()) >= 80
    predicted_log_odds = math.log(
        body["fraud_probability"] / (1 - body["fraud_probability"])
    )
    explained_log_odds = StubPredictor.BASE_LOG_ODDS + math.fsum(
        body["shap"].values()
    )
    assert explained_log_odds == pytest.approx(predicted_log_odds, abs=0.01)


def test_predict_returns_safe_result_below_threshold(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_SAFE_001",
            "features": raw_features_factory(),
        },
    )

    assert response.status_code == 200
    assert response.json()["is_fraud"] is False
    assert response.json()["fraud_probability"] < 0.2


def test_compound_transaction_risk_increases_probability(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    client = TestClient(create_app())
    safe = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_COMPARE_SAFE",
            "features": raw_features_factory(
                Customer_rooting_jailbreak_indicator=0,
                Customer_increase_atm_limit=0,
                Account_indicator_release_limit_excess=0,
                Another_Person_Account=0,
                Unused_account_status=0,
                Transaction_Datetime="2025-01-12 13:04:05",
                Number_of_transaction_with_the_account=20,
                Transaction_history_with_the_account=10,
            ),
        },
    ).json()
    risky = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_COMPARE_RISKY",
            "features": raw_features_factory(
                Transaction_Amount=12_000_000,
                Account_balance=5_000_000,
                Account_amount_daily_limit=5_000_000,
                Account_one_month_max_amount=3_000_000,
                Account_one_month_std_dev=100_000,
                Customer_VPN_Indicator=1,
                Unused_terminal_status=1,
                Recipient_account_suspend_status=1,
                Distance=300.0,
                **{"Time Difference": "0 days 00:30:00"},
            ),
        },
    ).json()

    assert safe["is_fraud"] is False
    assert risky["is_fraud"] is True
    assert risky["fraud_probability"] > safe["fraud_probability"] + 0.7
    assert risky["shap"]["Recipient_account_suspend_status"] > 0.6
    assert risky["shap"]["Distance"] > 0


def test_stub_predictor_reads_serving_environment(
    monkeypatch: pytest.MonkeyPatch,
    raw_features_factory: RawFeaturesFactory,
) -> None:
    monkeypatch.setenv("ML_FRAUD_THRESHOLD", "0.98")
    monkeypatch.setenv("ML_MODEL_NAME", "demo-model")
    monkeypatch.setenv("ML_MODEL_VERSION", "3")
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_000001",
            "features": raw_features_factory(
                Transaction_Amount=50_000_000,
                Account_one_month_std_dev=100_000,
                Customer_VPN_Indicator=1,
                Unused_terminal_status=1,
            ),
        },
    )

    assert response.status_code == 200
    # 제거된 과거 환경변수는 무시하고 Stub 자체 판정 기준을 사용한다.
    assert response.json()["is_fraud"] is True
    assert response.json()["model_name"] == "demo-model"
    assert response.json()["model_version"] == "3"


def test_stub_predictor_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="0과 1 사이"):
        StubPredictor(threshold=1.1)


def test_predict_rejects_missing_feature_container() -> None:
    client = TestClient(create_app())

    response = client.post("/predict", json={"transaction_id": "TEST_000001"})

    assert response.status_code == 422


def test_predict_rejects_partial_transaction_features(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    client = TestClient(create_app())
    features = raw_features_factory()
    del features["Location"]

    response = client.post(
        "/predict",
        json={"transaction_id": "TEST_000001", "features": features},
    )

    assert response.status_code == 422
    assert "Location" in response.text


def test_predict_rejects_unknown_features(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_DRAFT_001",
            "features": raw_features_factory(new_transaction_field="draft-value"),
        },
    )

    assert response.status_code == 422
    assert "new_transaction_field" in response.text


def test_predict_rejects_training_label_and_identifiers(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_000001",
            "features": raw_features_factory(Is_Fraud=1, ID="TEST_000001"),
        },
    )

    assert response.status_code == 422
    assert "Is_Fraud" in response.text
    assert "ID" in response.text


def test_predict_rejects_value_that_cannot_be_preprocessed(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_BAD_LOCATION",
            "features": raw_features_factory(
                Location="전북특별자치도 남원시 도통동 99.0 127.0"
            ),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "location_latitude must be in [33, 39]"


def test_current_model_contract_contains_54_raw_columns() -> None:
    assert len(MODEL_INPUT_COLUMNS) == 54
    assert len(MODEL_INPUT_COLUMNS) == len(set(MODEL_INPUT_COLUMNS))
