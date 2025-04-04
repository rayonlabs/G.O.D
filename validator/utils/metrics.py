import time
from typing import Callable

from fastapi import Request
from fastapi import Response
from opentelemetry import metrics

from validator.utils.logging import get_logger


logger = get_logger(__name__)

meter = metrics.get_meter("validator_api")

http_requests_total = meter.create_counter(name="http_requests_total", description="Total number of HTTP requests", unit="1")

http_request_duration_seconds = meter.create_histogram(
    name="http_request_duration_seconds", description="HTTP request duration in seconds", unit="s"
)

http_requests_in_progress = meter.create_up_down_counter(
    name="http_requests_in_progress", description="Number of HTTP requests in progress", unit="1"
)


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    start_time = time.time()

    http_requests_in_progress.add(1, {"method": request.method, "path": request.url.path})

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise e
    finally:
        http_requests_in_progress.add(-1, {"method": request.method, "path": request.url.path})

        duration = time.time() - start_time

        labels = {"method": request.method, "path": request.url.path, "status": str(status_code)}

        logger.info(f"Metrics: {labels}")
        logger.info(f"Duration: {duration}")
        http_requests_total.add(1, labels)
        http_request_duration_seconds.record(duration, labels)

    return response
