#!/bin/sh
set -eu

: "${MLFLOW_DATABASE_URI:?MLFLOW_DATABASE_URI is required}"
: "${MLFLOW_ADMIN_USERNAME:?MLFLOW_ADMIN_USERNAME is required}"
: "${MLFLOW_ADMIN_PASSWORD:?MLFLOW_ADMIN_PASSWORD is required}"
: "${MLFLOW_FLASK_SERVER_SECRET_KEY:?MLFLOW_FLASK_SERVER_SECRET_KEY is required}"

cat > /tmp/mlflow-basic-auth.ini <<EOF
[mlflow]
database_uri = ${MLFLOW_DATABASE_URI}
admin_username = ${MLFLOW_ADMIN_USERNAME}
admin_password = ${MLFLOW_ADMIN_PASSWORD}
default_permission = EDIT
grant_default_workspace_access = true
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF

export MLFLOW_AUTH_CONFIG_PATH=/tmp/mlflow-basic-auth.ini

exec "$@"
