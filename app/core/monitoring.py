from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Callable

from fastapi import Request, Response


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("lstm_stock_api")


@dataclass
class MetricsSnapshot:
    total_requests: int
    total_errors: int
    average_response_time_ms: float


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._total_requests = 0
        self._total_errors = 0
        self._total_response_time_ms = 0.0

    def record(self, duration_ms: float, is_error: bool) -> None:
        with self._lock:
            self._total_requests += 1
            self._total_response_time_ms += duration_ms
            if is_error:
                self._total_errors += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            average = (
                self._total_response_time_ms / self._total_requests
                if self._total_requests
                else 0.0
            )
            return MetricsSnapshot(
                total_requests=self._total_requests,
                total_errors=self._total_errors,
                average_response_time_ms=round(average, 2),
            )


metrics_store = MetricsStore()


async def monitoring_middleware(
    request: Request, call_next: Callable[[Request], Response]
) -> Response:
    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - started_at) * 1000
        metrics_store.record(duration_ms, status_code >= 500)
        logger.info(
            "request path=%s method=%s status=%s duration_ms=%.2f",
            request.url.path,
            request.method,
            status_code,
            duration_ms,
        )
