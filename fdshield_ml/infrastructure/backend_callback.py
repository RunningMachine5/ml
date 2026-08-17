"""Cloud Run Training Job 결과를 Backend에 전달한다."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CALLBACK_MAX_ATTEMPTS = 3
CALLBACK_RETRY_DELAYS_SECONDS = (1.0, 2.0)


class TrainingResultNotificationError(RuntimeError):
    """Backend에 학습 결과를 기록하지 못했을 때 발생한다."""


@dataclass(frozen=True)
class TrainingCallbackConfig:
    """전체 학습 설정이 잘못돼도 실패 콜백에 필요한 최소 설정."""

    backend_training_run_id: int | None = None
    result_callback_url: str = ""
    result_callback_token: str = ""
    cloud_run_execution_name: str = ""

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> TrainingCallbackConfig:
        backend_training_run_id_value = environ.get(
            "BACKEND_TRAINING_RUN_ID", ""
        ).strip()
        callback_url = environ.get("TRAINING_RESULT_CALLBACK_URL", "").strip()
        callback_token = environ.get("TRAINING_RESULT_CALLBACK_TOKEN", "").strip()
        cloud_run_execution_name = environ.get("CLOUD_RUN_EXECUTION", "").strip()

        if not backend_training_run_id_value:
            return cls(cloud_run_execution_name=cloud_run_execution_name)

        try:
            backend_training_run_id = int(backend_training_run_id_value)
        except ValueError as exc:
            raise ValueError("BACKEND_TRAINING_RUN_ID must be an integer") from exc
        if backend_training_run_id < 1:
            raise ValueError("BACKEND_TRAINING_RUN_ID must be positive")
        if not callback_url or not callback_token:
            raise ValueError(
                "TRAINING_RESULT_CALLBACK_URL and TOKEN are required with "
                "BACKEND_TRAINING_RUN_ID"
            )

        return cls(
            backend_training_run_id=backend_training_run_id,
            result_callback_url=callback_url,
            result_callback_token=callback_token,
            cloud_run_execution_name=cloud_run_execution_name,
        )


def notify_training_result(
    config: TrainingCallbackConfig,
    payload: dict[str, object],
    *,
    opener: Callable[..., object] = urlopen,
    sleeper: Callable[[float], None] = sleep,
) -> None:
    """Backend에 최소 학습 결과를 기록하고 일시적 실패는 재시도한다."""

    if config.backend_training_run_id is None:
        return
    url = config.result_callback_url.format(
        training_run_id=config.backend_training_run_id
    )
    encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):
        request = Request(
            url,
            data=encoded_payload,
            headers={
                "Content-Type": "application/json",
                "X-MLOps-Admin-Token": config.result_callback_token,
            },
            method="POST",
        )
        try:
            with opener(request, timeout=20) as response:  # type: ignore[attr-defined]
                status = response.status  # type: ignore[attr-defined]
                if 200 <= status < 300:
                    return
                if not _is_retryable_callback_status(status):
                    raise TrainingResultNotificationError(
                        f"Backend callback returned HTTP {status}."
                    )
        except HTTPError as exc:
            if not _is_retryable_callback_status(exc.code):
                raise TrainingResultNotificationError(
                    f"Backend callback returned HTTP {exc.code}."
                ) from exc
            last_error: BaseException = exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
        else:
            last_error = TrainingResultNotificationError(
                "Backend callback returned a retryable HTTP status."
            )

        if attempt < CALLBACK_MAX_ATTEMPTS:
            sleeper(CALLBACK_RETRY_DELAYS_SECONDS[attempt - 1])

    raise TrainingResultNotificationError(
        "Failed to record the training result in Backend after retries."
    ) from last_error


def _is_retryable_callback_status(status: int) -> bool:
    return status in {408, 429} or status >= 500


def with_cloud_run_execution(
    config: TrainingCallbackConfig,
    payload: dict[str, object],
) -> dict[str, object]:
    """Cloud Run 실행 이름을 설정된 경우에만 Callback payload에 추가한다."""

    if not config.cloud_run_execution_name:
        return payload
    return {
        **payload,
        "cloud_run_execution_name": config.cloud_run_execution_name,
    }
