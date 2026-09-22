from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ProxyError(Exception):
    status_code: int = 500
    error_type: str = "proxy_error"
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code
        self.provider = provider
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "message": self.message,
            "type": self.error_type,
            "code": self.code,
        }
        if self.provider:
            error["provider"] = self.provider
        if self.details:
            error["details"] = self.details
        return {"error": error}


class InvalidRequestError(ProxyError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request"


class AuthenticationError(ProxyError):
    status_code = 401
    error_type = "authentication_error"
    code = "invalid_api_key"


class ConfigurationError(ProxyError):
    status_code = 500
    error_type = "configuration_error"
    code = "misconfigured"


class ProviderUnavailableError(ProxyError):
    status_code = 503
    error_type = "provider_unavailable"
    code = "backend_unreachable"


class ProviderTimeoutError(ProxyError):
    status_code = 504
    error_type = "provider_timeout"
    code = "backend_timeout"


class CompressionError(ProxyError):
    status_code = 500
    error_type = "compression_error"
    code = "compression_failed"


class UpstreamError(ProxyError):
    status_code = 502
    error_type = "upstream_error"
    code = "backend_error"

    def __init__(
        self,
        message: str,
        *,
        upstream_status: int,
        provider: str,
        body: Any = None,
    ) -> None:
        super().__init__(
            message,
            status_code=502 if upstream_status >= 500 else upstream_status,
            provider=provider,
            details={"upstream_status": upstream_status, "upstream_body": body},
        )
        self.upstream_status = upstream_status


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProxyError)
    async def _proxy_error_handler(_: Request, exc: ProxyError) -> JSONResponse:
        log = logger.warning if exc.status_code < 500 else logger.error
        log("proxy error (%s): %s", exc.status_code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "Request body failed validation.",
                    "type": "invalid_request_error",
                    "code": "validation_error",
                    "details": {"errors": jsonable_encoder(exc.errors())},
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal proxy error.",
                    "type": "proxy_error",
                    "code": "internal_error",
                }
            },
        )
