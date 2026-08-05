"""모델 서빙 스켈레톤의 HTTP 계약 테스트."""

from fastapi.testclient import TestClient
import pytest

from fdshield_ml.serving.app import create_app
from fdshield_ml.serving.feature_contract import MODEL_INPUT_COLUMNS
from fdshield_ml.serving.predictor import StubPredictor


def _raw_features(**overrides: object) -> dict[str, object]:
    """실제 55개 입력 컬럼을 가진 최소 테스트 거래를 만든다."""

    features: dict[str, object] = {column: None for column in MODEL_INPUT_COLUMNS}
    features.update(
        {
            "Transaction_Amount": 100_000,
            "Account_one_month_std_dev": 100_000,
            "Customer_VPN_Indicator": 0,
            "Unused_terminal_status": 0,
            "Recipient_account_suspend_status": 0,
            "Transaction_Failure_Status": 0,
            **overrides,
        }
    )
    return features


def test_health_and_readiness() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_predict_returns_deterministic_stub_contract() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_000001",
            "features": _raw_features(
                Transaction_Amount=-50_000_000,
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
            "Transaction_Failure_Status": 0.0,
        },
        "model_name": "fdshield-rule-based-stub",
        "model_version": "0",
    }


def test_predict_returns_safe_result_below_threshold() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_SAFE_001",
            "features": _raw_features(),
        },
    )

    assert response.status_code == 200
    assert response.json()["is_fraud"] is False
    assert response.json()["fraud_probability"] == 0.1


def test_stub_predictor_reads_serving_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_FRAUD_THRESHOLD", "0.98")
    monkeypatch.setenv("ML_MODEL_NAME", "demo-model")
    monkeypatch.setenv("ML_MODEL_VERSION", "3")
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_000001",
            "features": _raw_features(
                Transaction_Amount=-50_000_000,
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


def test_predict_accepts_partial_transaction_features() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_000001",
            "features": {"Transaction_Amount": 850_000},
        },
    )

    assert response.status_code == 200
    assert response.json()["fraud_probability"] == 0.1
    assert response.json()["shap"] == {"Transaction_Amount": 0.05}


def test_predict_accepts_unknown_temporary_features() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_DRAFT_001",
            "features": {"new_transaction_field": "draft-value"},
        },
    )

    assert response.status_code == 200
    assert response.json()["fraud_probability"] == 0.05
    assert response.json()["shap"] == {}


def test_predict_rejects_training_label_and_identifiers() -> None:
    client = TestClient(create_app())

    features = _raw_features(Fraud_Type="m", ID="TEST_000001")
    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_000001",
            "features": features,
        },
    )

    assert response.status_code == 422


def test_current_model_contract_contains_55_raw_columns() -> None:
    assert len(MODEL_INPUT_COLUMNS) == 55
