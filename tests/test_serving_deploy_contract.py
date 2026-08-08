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
