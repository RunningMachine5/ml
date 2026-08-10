from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_build_smokes_exact_mlflow_model_with_secret_env() -> None:
    cloud_build = (ROOT / "cloudbuild.serving.yaml").read_text(encoding="utf-8")

    assert "smoke-serving-image" in cloud_build
    assert "secretEnv:" in cloud_build
    assert "MLFLOW_TRACKING_USERNAME" in cloud_build
    assert "MLFLOW_TRACKING_PASSWORD" in cloud_build
    assert '"http://127.0.0.1:8080/ready"' in cloud_build
    assert '"http://127.0.0.1:8080/predict"' in cloud_build
    assert "prediction.get(\"model_name\")" in cloud_build
    assert "prediction.get(\"model_version\")" in cloud_build


def test_github_runner_does_not_call_internal_cloud_run_url() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "deploy-serving.yml"
    ).read_text(encoding="utf-8")

    assert "Issue Cloud Run identity token for smoke test" not in workflow
    assert "Smoke test tagged revision" not in workflow
    assert "Verify tagged revision configuration" in workflow
    assert "Cloud Build-smoked digest and exact model config" in workflow
    assert '--remove-env-vars="ML_FRAUD_THRESHOLD"' in workflow
    assert 'if "ML_FRAUD_THRESHOLD" in environment:' in workflow
    assert "status.traffic[0]" not in workflow
    assert "revision_percent != {expected_revision: 100}" in workflow


def test_github_actions_preserves_active_approved_model_version() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "deploy-serving.yml"
    ).read_text(encoding="utf-8")

    assert "${{ vars.ML_MODEL_VERSION }}" not in workflow
    assert "REPOSITORY_MODEL_VERSION" not in workflow
    assert "DISPATCH_MODEL_VERSION" not in workflow
    assert "inputs:\n      model_version:" not in workflow
    assert 'gcloud run revisions describe "$active_revision"' in workflow
    assert "exactly one approved" in workflow
    assert 'github_env.write(f"MODEL_VERSION={model_version}\\n")' in workflow


def test_training_deploy_removes_legacy_promotion_environment() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "deploy-training.yml"
    ).read_text(encoding="utf-8")

    assert '--remove-env-vars="MLFLOW_AUTO_PROMOTE"' in workflow
    assert "Legacy automatic promotion environment is absent." in workflow


def test_bundled_model_is_in_serving_image_and_ci_paths() -> None:
    dockerfile = (ROOT / "Dockerfile.serving").read_text(encoding="utf-8")
    compose = (ROOT / "compose.serving.yml").read_text(encoding="utf-8")
    ci_workflow = (
        ROOT / ".github" / "workflows" / "ci-serving.yml"
    ).read_text(encoding="utf-8")
    deploy_workflow = (
        ROOT / ".github" / "workflows" / "deploy-serving.yml"
    ).read_text(encoding="utf-8")

    assert "COPY --chown=appuser:appuser models ./models" in dockerfile
    assert "ML_PREDICTOR_MODE: ${ML_PREDICTOR_MODE:-local}" in compose
    assert "fdshield-rule-based-stub" not in compose
    assert '"models/**"' in ci_workflow
    assert '"models/**"' in deploy_workflow
    assert "Verify bundled model readiness" in ci_workflow
