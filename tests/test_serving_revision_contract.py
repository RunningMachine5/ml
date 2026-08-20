from __future__ import annotations

from copy import deepcopy

import pytest

from deploy.serving_revision_contract import (
    ServingRevisionContractError,
    resolve_tag_target,
    revision_traffic_percent,
    validate_mlflow_tracking_uri,
    validate_zero_traffic_revision,
)


TAG = "model-v17"
APPROVED_REVISION = "fdshield-00001-approved"
STAGED_REVISION = "fdshield-00002-staged"
DIGEST = "asia-northeast3-docker.pkg.dev/project/fdshield/ml-serving@sha256:abc"


def _service() -> dict[str, object]:
    return {
        "metadata": {"generation": 9},
        "status": {
            "observedGeneration": 9,
            "conditions": [{"type": "Ready", "status": "True"}],
            "url": "https://fdshield.run.app",
            "latestCreatedRevisionName": STAGED_REVISION,
            "latestReadyRevisionName": STAGED_REVISION,
            "traffic": [
                {
                    "revisionName": APPROVED_REVISION,
                    "percent": 100,
                    "url": "https://fdshield.run.app",
                },
                {
                    "revisionName": STAGED_REVISION,
                    "percent": 0,
                    "tag": TAG,
                    "url": "https://model-v17---fdshield.run.app",
                },
            ],
        }
    }


def _revision() -> dict[str, object]:
    return {
        "metadata": {"name": STAGED_REVISION, "generation": 1},
        "spec": {
            "containers": [
                {
                    "image": DIGEST,
                    "env": [
                        {"name": "ML_PREDICTOR_MODE", "value": "mlflow"},
                        {
                            "name": "ML_MODEL_NAME",
                            "value": "fdshield-fraud-detector-v2",
                        },
                        {"name": "ML_MODEL_VERSION", "value": "17"},
                    ],
                }
            ]
        },
        "status": {
            "observedGeneration": 1,
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


def _validate(
    service: dict[str, object] | None = None,
    revision: dict[str, object] | None = None,
) -> None:
    validate_zero_traffic_revision(
        service=service or _service(),
        revision=revision or _revision(),
        expected_revision=STAGED_REVISION,
        expected_tag=TAG,
        expected_digest=DIGEST,
        expected_mode="mlflow",
        expected_model_name="fdshield-fraud-detector-v2",
        expected_model_version="17",
    )


def test_resolve_model_version_tag_and_zero_traffic() -> None:
    revision, url = resolve_tag_target(_service(), TAG)

    assert revision == STAGED_REVISION
    assert url == "https://model-v17---fdshield.run.app"
    assert revision_traffic_percent(_service(), STAGED_REVISION) == 0


def test_missing_model_version_tag_is_available_for_first_deployment() -> None:
    revision, url = resolve_tag_target(_service(), "model-v18")

    assert revision == ""
    assert url == ""


def test_exact_existing_model_tag_is_safe_to_reuse_idempotently() -> None:
    _validate()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("latest-created", "not the latest created revision"),
        ("latest-ready", "not the latest Ready revision"),
        ("service-reconciling", "service reconciliation is still in progress"),
        ("service-not-ready", "service Ready condition is not True"),
        ("revision-reconciling", "revision reconciliation is still in progress"),
        ("revision-not-ready", "revision Ready condition is not True"),
        ("tag-target", "tag targets a different revision"),
        ("digest", "verified image digest"),
        ("model-version", "ML_MODEL_VERSION"),
        ("legacy-threshold", "legacy ML_FRAUD_THRESHOLD"),
        ("production-traffic", "has production traffic"),
    ],
)
def test_existing_model_tag_collision_is_rejected(
    mutation: str, message: str
) -> None:
    service = deepcopy(_service())
    revision = deepcopy(_revision())
    status = service["status"]
    containers = revision["spec"]["containers"]

    if mutation == "latest-created":
        status["latestCreatedRevisionName"] = "fdshield-00003-newer"
    elif mutation == "latest-ready":
        status["latestReadyRevisionName"] = APPROVED_REVISION
    elif mutation == "service-reconciling":
        status["observedGeneration"] = 8
    elif mutation == "service-not-ready":
        status["conditions"][0]["status"] = "Unknown"
    elif mutation == "revision-reconciling":
        revision["status"]["observedGeneration"] = 0
    elif mutation == "revision-not-ready":
        revision["status"]["conditions"][0]["status"] = "False"
    elif mutation == "tag-target":
        status["traffic"][1]["revisionName"] = "fdshield-00003-other"
    elif mutation == "digest":
        containers[0]["image"] = f"{DIGEST}-different"
    elif mutation == "model-version":
        containers[0]["env"][2]["value"] = "18"
    elif mutation == "legacy-threshold":
        containers[0]["env"].append(
            {"name": "ML_FRAUD_THRESHOLD", "value": "0.5"}
        )
    elif mutation == "production-traffic":
        status["traffic"][0]["percent"] = 99
        status["traffic"][1]["percent"] = 1

    with pytest.raises(ServingRevisionContractError, match=message):
        _validate(service, revision)


def test_duplicate_model_version_tag_targets_are_rejected() -> None:
    service = _service()
    duplicate = dict(service["status"]["traffic"][1])
    duplicate["revisionName"] = "fdshield-00003-other"
    service["status"]["traffic"].append(duplicate)

    with pytest.raises(ServingRevisionContractError, match="more than one revision"):
        resolve_tag_target(service, TAG)


@pytest.mark.parametrize(
    "value",
    [
        "https://mlflow.internal.example.com",
        "http://mlflow:5000/api/2.0/mlflow",
        "https://mlflow.example.com:8443/base/path",
    ],
)
def test_plain_http_mlflow_tracking_uri_is_accepted(value: str) -> None:
    assert validate_mlflow_tracking_uri(value) == value


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("https://user:password@mlflow.example.com", "credentials"),
        ("https://mlflow.example.com?token=secret", "query"),
        ("https://mlflow.example.com?", "query"),
        ("https://mlflow.example.com/#fragment", "fragment"),
        ("https://mlflow.example.com/#", "fragment"),
        ("ftp://mlflow.example.com", "scheme and hostname"),
        ("https://mlflow.example.com:bad", "not a valid"),
        ("https://mlflow.example.com/path,other=value", "one plain"),
        (" https://mlflow.example.com", "one plain"),
    ],
)
def test_unsafe_mlflow_tracking_uri_is_rejected(
    value: str, message: str
) -> None:
    with pytest.raises(ServingRevisionContractError, match=message):
        validate_mlflow_tracking_uri(value)
