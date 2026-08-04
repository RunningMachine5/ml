"""모델 서빙 스켈레톤의 HTTP 계약 테스트."""

from fastapi.testclient import TestClient

from fdshield_ml.serving.app import create_app


def test_health_and_readiness() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_predict_returns_stub_contract() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "user_id": "xxx_001",
            "features": {
                "transaction_time": "2026-08-04T15:00:00+09:00",
                "amount": 850_000,
                "user_amount_std_dev": 120_000,
                "payment_method": "CARD",
                "merchant_category": "VEHICLES",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "xxx_001",
        "is_fraud": False,
        "fraud_probability": 0.1,
        "shap": {},
        "model_name": "stub-model",
        "model_version": "0",
    }


def test_predict_rejects_missing_feature_container() -> None:
    client = TestClient(create_app())

    response = client.post("/predict", json={"user_id": "xxx_001"})

    assert response.status_code == 422


def test_predict_rejects_incomplete_transaction_features() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "user_id": "xxx_001",
            "features": {
                "amount": 850_000,
                "payment_method": "CARD",
            },
        },
    )

    assert response.status_code == 422


def test_predict_rejects_invalid_numeric_features() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={
            "user_id": "xxx_001",
            "features": {
                "transaction_time": "2026-08-04T15:00:00+09:00",
                "amount": -1,
                "user_amount_std_dev": 0,
                "payment_method": "CARD",
                "merchant_category": "VEHICLES",
            },
        },
    )

    assert response.status_code == 422
