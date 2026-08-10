"""모델 Serving HTTP 계약 테스트."""

from collections.abc import Callable

import numpy as np
from fastapi.testclient import TestClient

from fdshield_ml.common.feature_contract import MODEL_INPUT_COLUMNS
from fdshield_ml.serving.app import create_app
from fdshield_ml.serving.model_predictor import ModelPredictor

RawFeaturesFactory = Callable[..., dict[str, object]]


class FakeProbabilityModel:
    """HTTP 요청 검증에서 외부 모델 로딩만 격리하는 테스트 대역."""

    decision_threshold_ = 0.5

    def predict_proba(self, features: object) -> np.ndarray:
        assert features.shape == (1, 91)
        return np.asarray([[0.2, 0.8]])


def _client() -> TestClient:
    return TestClient(
        create_app(
            ModelPredictor(
                model=FakeProbabilityModel(),
                model_name="test-model",
                model_version="1",
            )
        )
    )


def test_health_and_readiness() -> None:
    client = _client()

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_predict_returns_model_contract(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    client = _client()

    response = client.post(
        "/predict",
        json={
            "transaction_id": "TEST_000001",
            "features": raw_features_factory(),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "transaction_id": "TEST_000001",
        "is_fraud": True,
        "fraud_probability": 0.8,
        "shap": {},
        "model_name": "test-model",
        "model_version": "1",
    }


def test_predict_rejects_missing_feature_container() -> None:
    response = _client().post("/predict", json={"transaction_id": "TEST_000001"})

    assert response.status_code == 422


def test_predict_rejects_partial_transaction_features(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    client = _client()
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
    response = _client().post(
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
    response = _client().post(
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
    response = _client().post(
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
