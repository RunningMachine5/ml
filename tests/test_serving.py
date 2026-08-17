"""정식 raw51 -> model79 Serving HTTP 계약 테스트."""

from collections.abc import Callable

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fdshield_ml import serving as serving_app
from fdshield_ml.config.preprocess_config import (
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
    SERVING_INPUT_COLUMNS,
)
from fdshield_ml.dto.predict_input import PredictInputDTO
from fdshield_ml.service.predict.predict_service import (
    PredictionServiceError,
    PredictService,
)
from fdshield_ml.serving import create_app

RawFeaturesFactory = Callable[..., dict[str, object]]


class FakeProbabilityModel:
    """HTTP 계약에서 외부 모델 로딩만 격리하는 model79 테스트 대역."""

    decision_threshold_ = 0.5

    def predict_proba(self, features: object) -> np.ndarray:
        assert features.shape == (1, 79)
        assert list(features.columns) == list(MODEL_FEATURE_COLUMNS)
        return np.asarray([[0.2, 0.8]])


def _client() -> TestClient:
    return TestClient(
        create_app(
            PredictService(
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


def test_lifespan_loads_model_service_before_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PredictService(
        model=FakeProbabilityModel(),
        model_name="startup-model",
        model_version="1",
    )
    monkeypatch.setattr(
        serving_app,
        "predict_service_from_environment",
        lambda: service,
    )

    with TestClient(create_app()) as client:
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.app.state.predict_service is service


def test_lifespan_fails_fast_when_model_loading_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_load() -> PredictService:
        raise PredictionServiceError("model load failed")

    monkeypatch.setattr(
        serving_app,
        "predict_service_from_environment",
        fail_to_load,
    )

    with (
        pytest.raises(PredictionServiceError, match="model load failed"),
        TestClient(create_app()),
    ):
        pass


def test_transition_predict_endpoint_is_removed() -> None:
    response = _client().post("/predict", json={})

    assert response.status_code == 404


def test_ml_predict_returns_official_flat_contract(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    response = _client().post(
        "/ml/predict",
        json={"transaction_id": 1001, **raw_features_factory()},
    )

    assert response.status_code == 200
    assert response.json() == {
        "transaction_id": 1001,
        "predict_result": 1,
        "predict_proba": 0.8,
        "shap_values": {},
        "model_name": "test-model",
        "model_version": "1",
    }


def test_ml_predict_rejects_missing_features() -> None:
    response = _client().post("/ml/predict", json={"transaction_id": 1001})

    assert response.status_code == 422


def test_ml_predict_rejects_partial_transaction_features(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    features = raw_features_factory()
    del features["distance"]

    response = _client().post(
        "/ml/predict",
        json={"transaction_id": 1001, **features},
    )

    assert response.status_code == 422
    assert "distance" in response.text


def test_ml_predict_rejects_unknown_features(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    response = _client().post(
        "/ml/predict",
        json={
            "transaction_id": 1002,
            **raw_features_factory(new_transaction_field="draft-value"),
        },
    )

    assert response.status_code == 422
    assert "new_transaction_field" in response.text


def test_ml_predict_rejects_training_label_and_identifiers(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    response = _client().post(
        "/ml/predict",
        json={
            "transaction_id": 1003,
            **raw_features_factory(is_fraud=1, customer_id=123),
        },
    )

    assert response.status_code == 422
    assert "is_fraud" in response.text
    assert "customer_id" in response.text


def test_ml_predict_rejects_invalid_duration(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    response = _client().post(
        "/ml/predict",
        json={
            "transaction_id": 1004,
            **raw_features_factory(time_difference="not-a-duration"),
        },
    )

    assert response.status_code == 422
    assert "time_difference" in response.text


def test_current_model_contract_contains_raw51_and_model79() -> None:
    assert len(MODEL_INPUT_COLUMNS) == len(set(MODEL_INPUT_COLUMNS)) == 51
    assert len(SERVING_INPUT_COLUMNS) == len(set(SERVING_INPUT_COLUMNS)) == 52
    assert len(MODEL_FEATURE_COLUMNS) == len(set(MODEL_FEATURE_COLUMNS)) == 79


def test_xgboost_shap_uses_best_iteration_range(
    raw_features_factory: RawFeaturesFactory,
) -> None:
    class RecordingBooster:
        iteration_range: tuple[int, int] | None = None

        def num_boosted_rounds(self) -> int:
            return 8

        def predict(
            self,
            matrix: object,
            *,
            pred_contribs: bool,
            iteration_range: tuple[int, int],
        ) -> np.ndarray:
            assert pred_contribs is True
            assert matrix.num_row() == 1  # type: ignore[attr-defined]
            self.iteration_range = iteration_range
            return np.arange(
                len(MODEL_FEATURE_COLUMNS) + 1,
                dtype="float64",
            ).reshape(1, -1)

    booster = RecordingBooster()

    class EarlyStoppedModel:
        decision_threshold_ = 0.5
        best_iteration = 2

        def predict_proba(self, features: object) -> np.ndarray:
            return np.asarray([[0.7, 0.3]])

        def get_booster(self) -> RecordingBooster:
            return booster

    service = PredictService(
        model=EarlyStoppedModel(),
        model_name="candidate",
        model_version="7",
    )

    result = service.predict(
        PredictInputDTO.model_validate(
            {
                "transaction_id": 1008,
                **raw_features_factory(),
            }
        )
    )

    assert len(result.shap_values) == 55
    assert result.shap_values["customer_age"] == 0.0
    assert result.shap_values["time_difference"] == 22.0
    assert result.shap_values["distance"] == 23.0
    assert result.shap_values["customer_gender"] == 97.0
    assert result.shap_values["access_medium"] == sum(map(float, range(71, 79)))
    assert booster.iteration_range == (0, 3)
