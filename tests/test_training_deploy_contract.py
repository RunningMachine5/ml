"""Training 이미지와 Cloud Run Job 배포 계약 테스트."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from deploy.validate_training_job_contract import (
    is_plain_mlflow_tracking_uri,
    validate_training_job_contract,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "expected_image": "asia-northeast3-docker.pkg.dev/project/repo/ml@sha256:abc",
    "expected_data_uri": "gs://private-bucket/train1.csv",
    "expected_model_name": "fdshield-fraud-detector-v2",
    "expected_experiment_name": "fdshield-binary-training",
    "expected_callback_url": (
        "https://api.example/mlops/training/runs/{training_run_id}/result"
    ),
    "expected_service_account": "fdshield-training@project.iam.gserviceaccount.com",
    "expected_mlflow_tracking_uri": "https://mlflow.example.com/api",
    "expected_mlflow_tracking_username_secret": "mlflow-tracking-username-secret",
    "expected_mlflow_tracking_password_secret": "mlflow-tracking-password-secret",
    "expected_callback_token_secret": "training-result-callback-token-secret",
}


def _secret_environment(name: str, secret_name: str) -> dict[str, Any]:
    return {
        "name": name,
        "valueFrom": {
            "secretKeyRef": {
                "name": secret_name,
                "key": "latest",
            }
        },
    }


def _valid_job() -> dict[str, Any]:
    environment = [
        {"name": "TRAINING_DATA_URI", "value": EXPECTED["expected_data_uri"]},
        {
            "name": "MLFLOW_REGISTERED_MODEL_NAME",
            "value": EXPECTED["expected_model_name"],
        },
        {
            "name": "MLFLOW_EXPERIMENT_NAME",
            "value": EXPECTED["expected_experiment_name"],
        },
        {
            "name": "TRAINING_RESULT_CALLBACK_URL",
            "value": EXPECTED["expected_callback_url"],
        },
        {
            "name": "MLFLOW_TRACKING_URI",
            "value": EXPECTED["expected_mlflow_tracking_uri"],
        },
        _secret_environment(
            "MLFLOW_TRACKING_USERNAME",
            EXPECTED["expected_mlflow_tracking_username_secret"],
        ),
        _secret_environment(
            "MLFLOW_TRACKING_PASSWORD",
            EXPECTED["expected_mlflow_tracking_password_secret"],
        ),
        _secret_environment(
            "TRAINING_RESULT_CALLBACK_TOKEN",
            EXPECTED["expected_callback_token_secret"],
        ),
    ]
    return {
        "spec": {
            "template": {
                "spec": {
                    "template": {
                        "spec": {
                            "serviceAccountName": EXPECTED[
                                "expected_service_account"
                            ],
                            "containers": [
                                {
                                    "image": EXPECTED["expected_image"],
                                    "env": environment,
                                }
                            ],
                        }
                    }
                }
            }
        }
    }


def test_training_job_contract_accepts_plain_uri_and_secret_credentials() -> None:
    assert validate_training_job_contract(_valid_job(), **EXPECTED) == []


def test_plain_mlflow_tracking_uri_validation() -> None:
    assert is_plain_mlflow_tracking_uri("https://mlflow.example.com:8443/api")
    assert is_plain_mlflow_tracking_uri("http://mlflow:5000")
    assert not is_plain_mlflow_tracking_uri(
        "https://user:password@mlflow.example.com"
    )
    assert not is_plain_mlflow_tracking_uri("https://mlflow.example.com?token=x")
    assert not is_plain_mlflow_tracking_uri("https://mlflow.example.com#fragment")
    assert not is_plain_mlflow_tracking_uri("https://mlflow.example.com,OTHER=x")


def test_training_job_contract_rejects_secret_reference_for_tracking_uri() -> None:
    job = _valid_job()
    environment = job["spec"]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]["env"]
    tracking_uri = next(
        item for item in environment if item["name"] == "MLFLOW_TRACKING_URI"
    )
    tracking_uri.clear()
    tracking_uri.update(
        _secret_environment("MLFLOW_TRACKING_URI", "legacy-uri-secret")
    )

    errors = validate_training_job_contract(job, **EXPECTED)

    assert any("MLFLOW_TRACKING_URI" in error for error in errors)
    assert "legacy-uri-secret" not in repr(errors)


def test_training_job_contract_rejects_unsafe_plain_tracking_uri() -> None:
    expected = {
        **EXPECTED,
        "expected_mlflow_tracking_uri": (
            "https://user:sensitive@mlflow.example.com?token=sensitive"
        ),
    }
    job = _valid_job()
    environment = job["spec"]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]["env"]
    tracking_uri = next(
        item for item in environment if item["name"] == "MLFLOW_TRACKING_URI"
    )
    tracking_uri["value"] = expected["expected_mlflow_tracking_uri"]

    errors = validate_training_job_contract(job, **expected)

    assert any("plain HTTP(S) URL" in error for error in errors)
    assert "sensitive" not in repr(errors)


def test_training_job_contract_rejects_command_and_args_override() -> None:
    job = _valid_job()
    container = job["spec"]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    container["command"] = ["python"]
    container["args"] = ["-m", "fdshield_ml.training.old_job"]

    errors = validate_training_job_contract(job, **EXPECTED)

    assert "container command override must be empty to use the image CMD" in errors
    assert "container args override must be empty to use the image CMD" in errors


def test_training_job_contract_rejects_plain_text_secrets_without_leaking_values() -> None:
    job = _valid_job()
    environment = job["spec"]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]["env"]
    secret = next(
        item
        for item in environment
        if item["name"] == "MLFLOW_TRACKING_PASSWORD"
    )
    secret.clear()
    secret.update({"name": "MLFLOW_TRACKING_PASSWORD", "value": "do-not-print-me"})

    errors = validate_training_job_contract(job, **EXPECTED)

    assert (
        "environment must use the configured Secret Manager reference at version "
        "latest: MLFLOW_TRACKING_PASSWORD"
        in errors
    )
    assert "do-not-print-me" not in repr(errors)


def test_training_job_contract_rejects_wrong_secret_name_without_leaking_it() -> None:
    job = _valid_job()
    environment = job["spec"]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]["env"]
    secret = next(
        item for item in environment if item["name"] == "MLFLOW_TRACKING_USERNAME"
    )
    secret["valueFrom"]["secretKeyRef"]["name"] = "unexpected-sensitive-name"

    errors = validate_training_job_contract(job, **EXPECTED)

    assert any("MLFLOW_TRACKING_USERNAME" in error for error in errors)
    assert "unexpected-sensitive-name" not in repr(errors)


def test_training_job_contract_rejects_non_latest_secret_key_without_leaking_it() -> None:
    job = _valid_job()
    environment = job["spec"]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]["env"]
    secret = next(
        item for item in environment if item["name"] == "TRAINING_RESULT_CALLBACK_TOKEN"
    )
    secret["valueFrom"]["secretKeyRef"]["key"] = "sensitive-version-7"

    errors = validate_training_job_contract(job, **EXPECTED)

    assert any("TRAINING_RESULT_CALLBACK_TOKEN" in error for error in errors)
    assert "sensitive-version-7" not in repr(errors)


def test_training_job_contract_rejects_wrong_identity_and_static_run_id() -> None:
    job = _valid_job()
    task_spec = job["spec"]["template"]["spec"]["template"]["spec"]
    task_spec["serviceAccountName"] = (
        "123456789-compute@developer.gserviceaccount.com"
    )
    task_spec["containers"][0]["env"].append(
        {"name": "BACKEND_TRAINING_RUN_ID", "value": "7"}
    )

    errors = validate_training_job_contract(job, **EXPECTED)

    assert any("service account does not match" in error for error in errors)
    assert any("default Compute Engine service account" in error for error in errors)
    assert any("BACKEND_TRAINING_RUN_ID" in error for error in errors)


def test_training_job_contract_mismatch_errors_do_not_include_actual_values() -> None:
    job = deepcopy(_valid_job())
    environment = job["spec"]["template"]["spec"]["template"]["spec"][
        "containers"
    ][0]["env"]
    data_uri = next(item for item in environment if item["name"] == "TRAINING_DATA_URI")
    data_uri["value"] = "gs://sensitive-bucket/private-training.csv"

    errors = validate_training_job_contract(job, **EXPECTED)

    assert any("TRAINING_DATA_URI" in error for error in errors)
    assert "sensitive-bucket" not in repr(errors)


def test_training_image_ci_and_cloud_build_smoke_the_real_entrypoint() -> None:
    dockerfile = (ROOT / "Dockerfile.training").read_text(encoding="utf-8")
    ci_workflow = (ROOT / ".github" / "workflows" / "ci-training.yml").read_text(
        encoding="utf-8"
    )
    cloud_build = (ROOT / "cloudbuild.training.yaml").read_text(encoding="utf-8")

    assert 'CMD ["python", "-m", "fdshield_ml.training_job"]' in dockerfile
    for deployment_file in (ci_workflow, cloud_build):
        assert "training_job_configuration_error" in deployment_file
        assert "entrypoint" in deployment_file.lower()
        assert 'exit_code" -ne 2' in deployment_file


def test_training_deploy_clears_overrides_and_validates_runtime_contract() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy-training.yml").read_text(
        encoding="utf-8"
    )

    assert '--command=""' in workflow
    assert '--args=""' in workflow
    assert '--service-account="$TRAINING_JOB_SERVICE_ACCOUNT"' in workflow
    assert "deploy/validate_training_job_contract.py" in workflow
    assert (
        "TRAINING_JOB_SERVICE_ACCOUNT: ${{ vars.TRAINING_JOB_SERVICE_ACCOUNT }}"
        in workflow
    )
    assert (
        "TRAINING_RESULT_CALLBACK_URL: ${{ vars.TRAINING_RESULT_CALLBACK_URL }}"
        in workflow
    )
    assert "MLFLOW_TRACKING_URI: ${{ vars.MLFLOW_TRACKING_URI }}" in workflow
    assert "MLFLOW_TRACKING_URI_SECRET" not in workflow
    assert (
        "MLFLOW_TRACKING_USERNAME_SECRET: "
        "${{ vars.MLFLOW_TRACKING_USERNAME_SECRET }}" in workflow
    )
    assert (
        "MLFLOW_TRACKING_PASSWORD_SECRET: "
        "${{ vars.MLFLOW_TRACKING_PASSWORD_SECRET }}" in workflow
    )
    assert (
        "TRAINING_RESULT_CALLBACK_TOKEN_SECRET: "
        "${{ vars.TRAINING_RESULT_CALLBACK_TOKEN_SECRET }}"
    ) in workflow
    assert "--update-secrets=" in workflow
    assert "BACKEND_TRAINING_RUN_ID" in workflow
    assert '--expected-mlflow-tracking-uri "$MLFLOW_TRACKING_URI"' in workflow
    assert "is_plain_mlflow_tracking_uri" in workflow
    assert "--remove-secrets=\"$secret_uri_env\"" in workflow
    assert "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI" in workflow
    assert "--update-secrets=\"MLFLOW_TRACKING_USERNAME=" in workflow
    assert (
        '--expected-mlflow-tracking-username-secret '
        '"$MLFLOW_TRACKING_USERNAME_SECRET"'
    ) in workflow
    assert (
        '--expected-mlflow-tracking-password-secret '
        '"$MLFLOW_TRACKING_PASSWORD_SECRET"'
    ) in workflow
    assert (
        '--expected-callback-token-secret '
        '"$TRAINING_RESULT_CALLBACK_TOKEN_SECRET"'
    ) in workflow
