"""모델 서빙 스켈레톤의 HTTP 계약 테스트."""

from collections.abc import Callable

from fastapi.testclient import TestClient
import pytest

from fdshield_ml.common.feature_contract import MODEL_INPUT_COLUMNS
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
    assert response.json() == {
        "transaction_id": "TEST_000001",
        "is_fraud": True,
        "fraud_probability": 0.95,
        "shap": {
            "Transaction_Amount": 0.35,
            "Account_one_month_std_dev": 0.25,
            "Customer_VPN_Indicator": 0.15,
            "Unused_terminal_status": 0.15,
            "Recipient_account_suspend_status": 0.0,
        },
        "model_name": "fdshield-rule-based-stub",
        "model_version": "0",
    }


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
    assert response.json()["fraud_probability"] == 0.1


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
    assert response.json()["is_fraud"] is False
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
