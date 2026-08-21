from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nginx_injects_admin_token_for_frontend_admin_routes() -> None:
    compose = (ROOT / "deploy" / "compose.mlflow.yml").read_text(encoding="utf-8")
    nginx = (ROOT / "deploy" / "nginx" / "mlflow.conf").read_text(encoding="utf-8")

    assert "/etc/nginx/templates/mlflow.conf.template:ro" in compose
    assert "MLOPS_ADMIN_TOKEN: ${MLOPS_ADMIN_TOKEN:?" in compose
    assert "NGINX_ENVSUBST_FILTER: ^MLOPS_ADMIN_TOKEN$" in compose
    assert nginx.count('proxy_set_header X-MLOps-Admin-Token "${MLOPS_ADMIN_TOKEN}";') == 4


def test_nginx_forwards_rule_pattern_statistics_to_backend() -> None:
    nginx = (ROOT / "deploy" / "nginx" / "mlflow.conf").read_text(encoding="utf-8")

    assert "location = /api/rule-pattern-statistics" in nginx
    assert "rewrite ^/api/rule-pattern-statistics$ /rule-pattern-statistics break;" in nginx

