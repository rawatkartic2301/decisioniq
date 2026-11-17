"""Middleware for request tracing, rate limiting, and PII-redacted structured logging."""

import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Callable, Dict, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from decisioniq.guardrails.pii_scrubber import scrub_pii

logger = logging.getLogger("decisioniq.api")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assigns or propagates X-Request-ID for every HTTP request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiter per client IP address.

    Allows `max_requests` per `window_seconds`.
    """

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # client_ip -> (count, window_start_timestamp)
        self._clients: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, time.time()))

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        count, window_start = self._clients[client_ip]
        if now - window_start > self.window_seconds:
            # Window expired, reset
            self._clients[client_ip] = (1, now)
        else:
            if count >= self.max_requests:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Rate limit exceeded ({self.max_requests} requests per {self.window_seconds}s). Try again later."
                    },
                    headers={"Retry-After": str(int(self.window_seconds - (now - window_start)))},
                )
            self._clients[client_ip] = (count + 1, window_start)

        return await call_next(request)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming and outgoing HTTP exchanges in structured JSON with PII scrubbing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "none")

        client_host = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        response = await call_next(request)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        status_code = response.status_code

        log_payload = {
            "timestamp": time.time(),
            "request_id": request_id,
            "client_ip": client_host,
            "method": method,
            "path": scrub_pii(path),
            "status_code": status_code,
            "duration_ms": duration_ms,
        }

        logger.info(json.dumps(log_payload))
        return response
