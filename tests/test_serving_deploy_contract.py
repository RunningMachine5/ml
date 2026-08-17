import json
from pathlib import Path

from fdshield_ml.config.preprocess_config import SERVING_INPUT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_build_smokes_exact_mlflow_model_with_secret_env() -> None:
    cloud_build = (ROOT / "cloudbuild.serving.yaml").read_text(encoding="utf-8")
    smoke_request = json.loads(
        (ROOT / "deploy" / "smoke-request.json").read_text(encoding="utf-8")
    )

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
    assert "!= 55" in cloud_build
    assert tuple(smoke_request) == SERVING_INPUT_COLUMNS


def test_github_runner_does_not_call_internal_cloud_run_url() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy-serving.yml").read_text(
        encoding="utf-8"
    )
    contract = (ROOT / "deploy" / "serving_revision_contract.py").read_text(
        encoding="utf-8"
    )

    assert "Issue Cloud Run identity token for smoke test" not in workflow
    assert "Smoke test tagged revision" not in workflow
    assert "Verify tagged revision configuration" in workflow
    assert "deploy/serving_revision_contract.py verify" in workflow
    assert 'gcloud run revisions describe "$NEW_REVISION"' in workflow
    assert '--remove-env-vars="ML_FRAUD_THRESHOLD"' in workflow
    assert 'if "ML_FRAUD_THRESHOLD" in environment:' in contract
    assert "validate_mlflow_tracking_uri" in workflow
    assert 'status.get("latestCreatedRevisionName")' in contract
    assert "_resource_reconciliation_errors(service" in contract
    assert "status.traffic[0]" not in workflow
    assert "revision_percent != {approved_revision: 100}" in workflow
    assert "latestCreatedRevisionName" not in workflow


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
    assert 'revision_tag="model-v${MODEL_VERSION}"' in workflow
    assert "model-${GITHUB_SHA::12}" not in workflow
    assert "Reuse exact staged revision or reject model tag collision" in workflow
    assert "Model version tags are immutable and will not be moved" in workflow
    assert "SERVING_DEPLOY_MODE=reuse" in workflow
    assert "gcloud run services update-traffic" not in workflow
    assert "Confirm production traffic was not changed" in workflow
    assert "Use the Backend approval and smoke" in workflow


def test_training_deploy_removes_legacy_promotion_environment() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy-training.yml").read_text(
        encoding="utf-8"
    )

    assert (
        '--remove-env-vars="MLFLOW_AUTO_PROMOTE,TRAINING_JOB_TYPE,TRAINING_MODE,'
        'TRAINING_TRANSACTIONS_URI,TRAINING_SPLIT_DATETIME,BACKEND_TRAINING_RUN_ID"'
    ) in workflow
    assert "TRAINING_DATA_URI: ${{ vars.TRAINING_DATA_URI }}" in workflow
    assert (
        "TRAINING_JOB_SERVICE_ACCOUNT: ${{ vars.TRAINING_JOB_SERVICE_ACCOUNT }}"
        in workflow
    )
    assert (
        "TRAINING_RESULT_CALLBACK_URL: ${{ vars.TRAINING_RESULT_CALLBACK_URL }}"
        in workflow
    )
    assert "REGISTERED_MODEL_NAME: fdshield-fraud-detector-v2" in workflow
    assert 'if [[ "$TRAINING_DATA_URI" != gs://*/*.csv ]]' in workflow
    assert (
        '--update-env-vars="TRAINING_DATA_URI=$TRAINING_DATA_URI,'
        "MLFLOW_EXPERIMENT_NAME=$MLFLOW_EXPERIMENT_NAME,"
        "MLFLOW_REGISTERED_MODEL_NAME=$REGISTERED_MODEL_NAME,"
        "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI,"
        'TRAINING_RESULT_CALLBACK_URL=$TRAINING_RESULT_CALLBACK_URL"'
    ) in workflow
    assert '--command=""' in workflow
    assert '--args=""' in workflow
    assert '--update-secrets="MLFLOW_TRACKING_USERNAME=' in workflow
    assert "MLFLOW_TRACKING_URI_SECRET" not in workflow
    assert "validate_training_job_contract.py" in workflow


def test_training_deploy_migrates_environment_binding_types_before_update() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy-training.yml").read_text(
        encoding="utf-8"
    )

    migration_step = "Migrate Cloud Run environment binding types"
    update_step = "Update Cloud Run Training Job image"
    assert migration_step in workflow
    assert '--remove-secrets="$secret_uri_env"' in workflow
    assert '--remove-env-vars="$literal_secret_envs"' in workflow
    assert 'item.get("name") == "MLFLOW_TRACKING_URI"' in workflow
    assert '"value" in item' in workflow
    assert '"valueFrom" not in item' in workflow
    assert workflow.index(migration_step) < workflow.index(update_step)


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
