"""Validate a new Cloud Run Serving revision before it receives traffic."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ServingRevisionContractError(ValueError):
    """Raised when a staged Cloud Run revision breaks the deployment contract."""


def validate_mlflow_tracking_uri(value: str) -> str:
    """Return a plain HTTP(S) MLflow URI or reject unsafe URL components."""

    if not value or value != value.strip() or any(
        character.isspace() or character == "," for character in value
    ):
        raise ServingRevisionContractError(
            "MLFLOW_TRACKING_URI must be one plain HTTP(S) URL"
        )

    try:
        parsed = urlsplit(value)
        # Accessing ``port`` also rejects malformed ports such as ``:abc``.
        _ = parsed.port
    except ValueError as exc:
        raise ServingRevisionContractError(
            "MLFLOW_TRACKING_URI is not a valid HTTP(S) URL"
        ) from exc

    errors: list[str] = []
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        errors.append("scheme and hostname are required")
    if parsed.username is not None or parsed.password is not None:
        errors.append("embedded credentials are forbidden")
    if "?" in value:
        errors.append("query parameters are forbidden")
    if "#" in value:
        errors.append("fragments are forbidden")
    if errors:
        raise ServingRevisionContractError(
            "MLFLOW_TRACKING_URI " + "; ".join(errors)
        )
    return value


def _status(service: Mapping[str, Any]) -> Mapping[str, Any]:
    status = service.get("status", {})
    return status if isinstance(status, Mapping) else {}


def _traffic_targets(service: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    targets = _status(service).get("traffic", [])
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        return []
    return [target for target in targets if isinstance(target, Mapping)]


def _target_revision(
    target: Mapping[str, Any], latest_ready_revision: str
) -> str:
    revision = target.get("revisionName", "")
    if revision:
        return str(revision)
    if target.get("latestRevision"):
        return latest_ready_revision
    return ""


def resolve_tag_target(
    service: Mapping[str, Any], expected_tag: str
) -> tuple[str, str]:
    """Return the single revision and URL selected by ``expected_tag``."""

    status = _status(service)
    latest_ready_revision = str(status.get("latestReadyRevisionName", ""))
    matches = [
        target
        for target in _traffic_targets(service)
        if target.get("tag") == expected_tag
    ]
    if len(matches) > 1:
        raise ServingRevisionContractError(
            f"tag {expected_tag!r} targets more than one revision"
        )
    if not matches:
        return "", ""

    revision = _target_revision(matches[0], latest_ready_revision)
    url = str(matches[0].get("url", ""))
    if not revision or not url:
        raise ServingRevisionContractError(
            f"tag {expected_tag!r} is missing its revision or URL"
        )
    return revision, url


def revision_traffic_percent(
    service: Mapping[str, Any], expected_revision: str
) -> int:
    """Return all production traffic assigned to one concrete revision."""

    latest_ready_revision = str(
        _status(service).get("latestReadyRevisionName", "")
    )
    return sum(
        int(target.get("percent", 0) or 0)
        for target in _traffic_targets(service)
        if _target_revision(target, latest_ready_revision) == expected_revision
    )


def _revision_container(revision: Mapping[str, Any]) -> Mapping[str, Any]:
    spec = revision.get("spec", {})
    if not isinstance(spec, Mapping):
        raise ServingRevisionContractError("revision has no spec")
    containers = spec.get("containers", [])
    if (
        not isinstance(containers, Sequence)
        or isinstance(containers, (str, bytes))
        or not containers
        or not isinstance(containers[0], Mapping)
    ):
        raise ServingRevisionContractError("revision has no container configuration")
    return containers[0]


def _resource_reconciliation_errors(
    resource: Mapping[str, Any], resource_name: str
) -> list[str]:
    """Validate the v1 generation and top-level Ready reconciliation contract."""

    metadata = resource.get("metadata", {})
    status = resource.get("status", {})
    if not isinstance(metadata, Mapping) or not isinstance(status, Mapping):
        return [f"{resource_name} reconciliation state is missing"]

    generation = metadata.get("generation")
    observed_generation = status.get("observedGeneration")
    errors: list[str] = []
    if generation is None or observed_generation is None:
        errors.append(f"{resource_name} generation state is missing")
    elif str(generation) != str(observed_generation):
        errors.append(f"{resource_name} reconciliation is still in progress")

    ready_conditions = [
        condition
        for condition in status.get("conditions", [])
        if isinstance(condition, Mapping) and condition.get("type") == "Ready"
    ]
    if len(ready_conditions) != 1 or str(
        ready_conditions[0].get("status", "")
    ).lower() != "true":
        errors.append(f"{resource_name} Ready condition is not True")
    return errors


def validate_zero_traffic_revision(
    *,
    service: Mapping[str, Any],
    revision: Mapping[str, Any],
    expected_revision: str,
    expected_tag: str,
    expected_digest: str,
    expected_mode: str,
    expected_model_name: str,
    expected_model_version: str,
) -> None:
    """Validate the Ready, tag, image, model environment and 0% contract."""

    errors: list[str] = []
    status = _status(service)
    errors.extend(_resource_reconciliation_errors(service, "service"))
    if status.get("latestCreatedRevisionName") != expected_revision:
        errors.append("new revision is not the latest created revision")
    if status.get("latestReadyRevisionName") != expected_revision:
        errors.append("new revision is not the latest Ready revision")

    try:
        tagged_revision, _ = resolve_tag_target(service, expected_tag)
    except ServingRevisionContractError as exc:
        errors.append(str(exc))
    else:
        if tagged_revision != expected_revision:
            errors.append("revision tag targets a different revision")

    metadata = revision.get("metadata", {})
    revision_name = metadata.get("name", "") if isinstance(metadata, Mapping) else ""
    if revision_name and revision_name != expected_revision:
        errors.append("described revision does not match the tagged revision")
    errors.extend(_resource_reconciliation_errors(revision, "revision"))

    try:
        container = _revision_container(revision)
    except ServingRevisionContractError as exc:
        errors.append(str(exc))
    else:
        if container.get("image") != expected_digest:
            errors.append("new revision does not use the verified image digest")

        env_items = container.get("env", [])
        environment = {
            item.get("name"): item.get("value")
            for item in env_items
            if isinstance(item, Mapping) and item.get("name")
        }
        expected_environment = {
            "ML_PREDICTOR_MODE": expected_mode,
            "ML_MODEL_NAME": expected_model_name,
            "ML_MODEL_VERSION": expected_model_version,
        }
        for name, value in expected_environment.items():
            if environment.get(name) != value:
                errors.append(f"{name} does not match the expected model")
        if "ML_FRAUD_THRESHOLD" in environment:
            errors.append("legacy ML_FRAUD_THRESHOLD remains")

    if revision_traffic_percent(service, expected_revision) != 0:
        errors.append("new revision already has production traffic")

    if errors:
        raise ServingRevisionContractError("; ".join(errors))


def _load_json(path: str) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ServingRevisionContractError(f"{path} must contain a JSON object")
    return value


def _append_github_env(path: str, values: Mapping[str, str]) -> None:
    with Path(path).open("a", encoding="utf-8") as github_env:
        for name, value in values.items():
            if any(char in value for char in "\r\n"):
                raise ServingRevisionContractError(
                    f"{name} cannot contain a line break"
                )
            github_env.write(f"{name}={value}\n")


def _resolve_command(args: argparse.Namespace) -> None:
    service = _load_json(args.service)
    revision, _ = resolve_tag_target(service, args.tag)
    if args.require_present and not revision:
        raise ServingRevisionContractError(
            f"tag {args.tag!r} was not created by the Cloud Run deployment"
        )
    _append_github_env(
        args.github_env,
        {
            args.revision_env: revision,
        },
    )
    if revision:
        print(f"Resolved {args.tag} to {revision}.")
    else:
        print(f"Tag {args.tag} is available for a new revision.")


def _verify_command(args: argparse.Namespace) -> None:
    validate_zero_traffic_revision(
        service=_load_json(args.service),
        revision=_load_json(args.revision),
        expected_revision=args.expected_revision,
        expected_tag=args.expected_tag,
        expected_digest=args.expected_digest,
        expected_mode=args.expected_mode,
        expected_model_name=args.expected_model_name,
        expected_model_version=args.expected_model_version,
    )
    print(
        "Tagged revision is the latest Ready revision and uses the verified "
        "digest, exact model configuration, and 0% production traffic."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--service", required=True)
    resolve.add_argument("--tag", required=True)
    resolve.add_argument("--github-env", required=True)
    resolve.add_argument("--revision-env", default="TAGGED_REVISION")
    resolve.add_argument("--require-present", action="store_true")
    resolve.set_defaults(handler=_resolve_command)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--service", required=True)
    verify.add_argument("--revision", required=True)
    verify.add_argument("--expected-revision", required=True)
    verify.add_argument("--expected-tag", required=True)
    verify.add_argument("--expected-digest", required=True)
    verify.add_argument("--expected-mode", required=True)
    verify.add_argument("--expected-model-name", required=True)
    verify.add_argument("--expected-model-version", required=True)
    verify.set_defaults(handler=_verify_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.handler(args)
    except ServingRevisionContractError as exc:
        print(f"::error::{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
