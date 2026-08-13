"""Cloud Run Training Job의 배포 후 런타임 계약을 검증한다.

검증 과정에서는 환경변수 값이나 Secret 이름을 출력하지 않는다. 배포 로그에는
누락되거나 잘못 연결된 설정 이름만 남긴다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

LEGACY_ENVIRONMENT_NAMES = frozenset(
    {
        "MLFLOW_AUTO_PROMOTE",
        "TRAINING_JOB_TYPE",
        "TRAINING_MODE",
        "TRAINING_TRANSACTIONS_URI",
        "TRAINING_SPLIT_DATETIME",
    }
)
PER_EXECUTION_ENVIRONMENT_NAMES = frozenset({"BACKEND_TRAINING_RUN_ID"})
SECRET_ENVIRONMENT_NAMES = (
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "TRAINING_RESULT_CALLBACK_TOKEN",
)


def is_plain_mlflow_tracking_uri(value: object) -> bool:
    """Return whether value is a safe literal HTTP(S) MLflow endpoint."""

    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if any(character.isspace() or character == "," for character in value):
        return False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.lower() in {"http", "https"}
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and "?" not in value
        and "#" not in value
    )


def _task_spec(job: Mapping[str, Any]) -> Mapping[str, Any]:
    """gcloud run jobs describe(v1)의 단일 Task spec을 반환한다."""

    value: Any = job
    for key in ("spec", "template", "spec", "template", "spec"):
        if not isinstance(value, Mapping):
            raise ValueError("Cloud Run Job JSON does not contain a task spec")
        value = value.get(key)
    if not isinstance(value, Mapping):
        raise ValueError("Cloud Run Job JSON does not contain a task spec")
    return value


def _environment_by_name(
    container: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    environment: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    raw_environment = container.get("env", [])
    if not isinstance(raw_environment, Sequence) or isinstance(
        raw_environment, (str, bytes)
    ):
        return environment, ["container env must be a list"]

    for item in raw_environment:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        if name in environment:
            duplicates.append(name)
        environment[name] = item
    return environment, duplicates


def _secret_reference_matches(
    item: Mapping[str, Any], *, expected_name: str
) -> bool:
    value_from = item.get("valueFrom")
    if not isinstance(value_from, Mapping):
        return False
    secret_ref = value_from.get("secretKeyRef")
    if not isinstance(secret_ref, Mapping):
        return False
    return (
        secret_ref.get("name") == expected_name
        and secret_ref.get("key") == "latest"
    )


def validate_training_job_contract(
    job: Mapping[str, Any],
    *,
    expected_image: str,
    expected_data_uri: str,
    expected_model_name: str,
    expected_experiment_name: str,
    expected_callback_url: str,
    expected_service_account: str,
    expected_mlflow_tracking_uri: str,
    expected_mlflow_tracking_username_secret: str,
    expected_mlflow_tracking_password_secret: str,
    expected_callback_token_secret: str,
) -> list[str]:
    """배포된 Job 계약의 위반 사항을 값 노출 없이 반환한다."""

    errors: list[str] = []
    try:
        task_spec = _task_spec(job)
    except ValueError as error:
        return [str(error)]

    containers = task_spec.get("containers", [])
    if not isinstance(containers, Sequence) or isinstance(containers, (str, bytes)):
        return ["Cloud Run Job containers must be a list"]
    if len(containers) != 1 or not isinstance(containers[0], Mapping):
        return ["Cloud Run Training Job must contain exactly one container"]
    container = containers[0]

    if container.get("image") != expected_image:
        errors.append("container image does not match the immutable deploy digest")
    if container.get("command"):
        errors.append("container command override must be empty to use the image CMD")
    if container.get("args"):
        errors.append("container args override must be empty to use the image CMD")

    service_account = task_spec.get("serviceAccountName")
    if service_account != expected_service_account:
        errors.append("Cloud Run Job service account does not match the expected identity")
    if isinstance(service_account, str) and service_account.endswith(
        "-compute@developer.gserviceaccount.com"
    ):
        errors.append("Cloud Run Job must not use the default Compute Engine service account")

    environment, duplicates = _environment_by_name(container)
    if duplicates:
        errors.append(
            "duplicate environment variables: " + ", ".join(sorted(set(duplicates)))
        )

    unexpected_legacy = sorted(LEGACY_ENVIRONMENT_NAMES & environment.keys())
    if unexpected_legacy:
        errors.append(
            "legacy training environment remains: " + ", ".join(unexpected_legacy)
        )
    unexpected_static = sorted(PER_EXECUTION_ENVIRONMENT_NAMES & environment.keys())
    if unexpected_static:
        errors.append(
            "per-execution environment must not be configured on the Job: "
            + ", ".join(unexpected_static)
        )

    expected_plain_environment = {
        "TRAINING_DATA_URI": expected_data_uri,
        "MLFLOW_REGISTERED_MODEL_NAME": expected_model_name,
        "MLFLOW_EXPERIMENT_NAME": expected_experiment_name,
        "MLFLOW_TRACKING_URI": expected_mlflow_tracking_uri,
        "TRAINING_RESULT_CALLBACK_URL": expected_callback_url,
    }
    for name, expected_value in expected_plain_environment.items():
        item = environment.get(name)
        if item is None:
            errors.append(f"required environment is missing: {name}")
        elif item.get("value") != expected_value:
            errors.append(
                "required environment does not match the deployment contract: "
                f"{name}"
            )

    if not is_plain_mlflow_tracking_uri(expected_mlflow_tracking_uri):
        errors.append(
            "MLFLOW_TRACKING_URI must be a plain HTTP(S) URL without credentials, "
            "query, fragment, whitespace, or commas"
        )

    expected_secret_names = {
        "MLFLOW_TRACKING_USERNAME": expected_mlflow_tracking_username_secret,
        "MLFLOW_TRACKING_PASSWORD": expected_mlflow_tracking_password_secret,
        "TRAINING_RESULT_CALLBACK_TOKEN": expected_callback_token_secret,
    }
    for name in SECRET_ENVIRONMENT_NAMES:
        item = environment.get(name)
        if item is None:
            errors.append(f"required Secret environment is missing: {name}")
        elif not _secret_reference_matches(
            item, expected_name=expected_secret_names[name]
        ):
            errors.append(
                "environment must use the configured Secret Manager reference "
                f"at version latest: {name}"
            )

    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_json", type=Path)
    parser.add_argument("--expected-image", required=True)
    parser.add_argument("--expected-data-uri", required=True)
    parser.add_argument("--expected-model-name", required=True)
    parser.add_argument("--expected-experiment-name", required=True)
    parser.add_argument("--expected-callback-url", required=True)
    parser.add_argument("--expected-service-account", required=True)
    parser.add_argument("--expected-mlflow-tracking-uri", required=True)
    parser.add_argument("--expected-mlflow-tracking-username-secret", required=True)
    parser.add_argument("--expected-mlflow-tracking-password-secret", required=True)
    parser.add_argument("--expected-callback-token-secret", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with args.job_json.open(encoding="utf-8") as stream:
        job = json.load(stream)
    if not isinstance(job, Mapping):
        print("::error::Cloud Run Job JSON root must be an object")
        return 1

    errors = validate_training_job_contract(
        job,
        expected_image=args.expected_image,
        expected_data_uri=args.expected_data_uri,
        expected_model_name=args.expected_model_name,
        expected_experiment_name=args.expected_experiment_name,
        expected_callback_url=args.expected_callback_url,
        expected_service_account=args.expected_service_account,
        expected_mlflow_tracking_uri=args.expected_mlflow_tracking_uri,
        expected_mlflow_tracking_username_secret=(
            args.expected_mlflow_tracking_username_secret
        ),
        expected_mlflow_tracking_password_secret=(
            args.expected_mlflow_tracking_password_secret
        ),
        expected_callback_token_secret=args.expected_callback_token_secret,
    )
    if errors:
        for error in errors:
            print(f"::error::{error}")
        return 1

    print("Cloud Run Training Job runtime contract verified without exposing values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
