FROM ghcr.io/mlflow/mlflow:v3.13.0-full

RUN pip install --no-cache-dir \
    "mlflow[auth]==3.13.0" \
    "psycopg2-binary>=2.9,<3"

COPY deploy/mlflow/docker-entrypoint.sh /usr/local/bin/fdshield-mlflow-entrypoint
RUN chmod +x /usr/local/bin/fdshield-mlflow-entrypoint

ENTRYPOINT ["/usr/local/bin/fdshield-mlflow-entrypoint"]
CMD ["mlflow", "server"]
