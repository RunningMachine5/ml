from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_build_smokes_exact_mlflow_model_with_secret_env() -> None:
    cloud_build = (ROOT / "cloudbuild.serving.yaml").read_text(encoding="utf-8")

    assert "smoke-serving-image" in cloud_build
    assert "secretEnv:" in cloud_build
    assert "MLFLOW_TRACKING_USERNAME" in cloud_build
    assert "MLFLOW_TRACKING_PASSWORD" in cloud_build
    assert '"http://127.0.0.1:8080/ready"' in cloud_build
    assert '"http://127.0.0.1:8080/ml/predict"' in cloud_build
    assert 'prediction.get("model_name")' in cloud_build
    assert 'prediction.get("model_version")' in cloud_build
    assert 'prediction.get("predict_result")' in cloud_build
    assert 'prediction.get("predict_proba")' in cloud_build
    assert 'prediction.get("shap_values", {})' in cloud_build


def test_github_runner_does_not_call_internal_cloud_run_url() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy-serving.yml").read_text(
        encoding="utf-8"
    )

    assert "Issue Cloud Run identity token for smoke test" not in workflow
    assert "Smoke test tagged revision" not in workflow
    assert "Verify tagged revision configuration" in workflow
    assert "Cloud Build-smoked digest and exact model config" in workflow
    assert '--remove-env-vars="ML_FRAUD_THRESHOLD"' in workflow
    assert 'if "ML_FRAUD_THRESHOLD" in environment:' in workflow
    assert "status.traffic[0]" not in workflow
    assert "staged_percent != 0" in workflow


def test_github_actions_stages_requested_model_without_automatic_traffic() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy-serving.yml").read_text(
        encoding="utf-8"
    )

    assert "push:" not in workflow
    assert "workflow_dispatch:" in workflow
    assert "model_name:" in workflow
    assert "model_version:" in workflow
    assert "REQUESTED_MODEL_NAME: ${{ inputs.model_name }}" in workflow
    assert "REQUESTED_MODEL_VERSION: ${{ inputs.model_version }}" in workflow
    assert 'gcloud run revisions describe "$active_revision"' in workflow
    assert "--no-traffic" in workflow
    assert "gcloud run services update-traffic" not in workflow
    assert "Confirm production traffic was not changed" in workflow
    assert "Use the Backend approval and smoke" in workflow


def test_training_deploy_removes_legacy_promotion_environment() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy-training.yml").read_text(
        encoding="utf-8"
    )

    assert (
        '--remove-env-vars="MLFLOW_AUTO_PROMOTE,TRAINING_JOB_TYPE,TRAINING_MODE,'
        'TRAINING_TRANSACTIONS_URI,TRAINING_SPLIT_DATETIME"'
    ) in workflow
    assert "TRAINING_DATA_URI: ${{ vars.TRAINING_DATA_URI }}" in workflow
    assert "REGISTERED_MODEL_NAME: fdshield-fraud-detector-v2" in workflow
    assert 'if [[ "$TRAINING_DATA_URI" != gs://*/*.csv ]]' in workflow
    assert (
        '--update-env-vars="TRAINING_DATA_URI=$TRAINING_DATA_URI,'
        'MLFLOW_REGISTERED_MODEL_NAME=$REGISTERED_MODEL_NAME"'
    ) in workflow
    assert "train1/model80 contract is active" in workflow


def test_bundled_model_is_in_serving_image_and_ci_paths() -> None:
    dockerfile = (ROOT / "Dockerfile.serving").read_text(encoding="utf-8")
    compose = (ROOT / "compose.serving.yml").read_text(encoding="utf-8")
    ci_workflow = (ROOT / ".github" / "workflows" / "ci-serving.yml").read_text(
        encoding="utf-8"
    )
    deploy_workflow = (ROOT / ".github" / "workflows" / "deploy-serving.yml").read_text(
        encoding="utf-8"
    )

    assert "COPY --chown=appuser:appuser models ./models" in dockerfile
    assert "ML_PREDICTOR_MODE: ${ML_PREDICTOR_MODE:-local}" in compose
    assert "fdshield-rule-based-stub" not in compose
    assert '"models/**"' in ci_workflow
    assert "workflow_dispatch:" in deploy_workflow
    assert "Verify bundled model readiness" in ci_workflow
