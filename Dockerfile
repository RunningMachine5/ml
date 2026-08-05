FROM ghcr.io/astral-sh/uv:0.11.24 AS uv

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY deploy/serving-requirements.txt ./deploy/serving-requirements.txt

# Stub 이미지에는 학습·MLflow 패키지를 넣지 않고 API 실행 의존성만 설치한다.
RUN uv pip install --system --no-cache \
    --requirement deploy/serving-requirements.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser fdshield_ml ./fdshield_ml

USER appuser

EXPOSE 8080

CMD ["python", "-m", "fdshield_ml.serve"]
