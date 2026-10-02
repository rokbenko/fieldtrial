"""The ASGI application: operator console, REST API v1 and static files."""

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from fieldtrial.api.schemas import API_VERSION
from fieldtrial.api.v1 import router as api_router
from fieldtrial.services import ConcurrencyError, ServiceError
from fieldtrial.services.registry import StudyRegistry, UnknownStudyError
from fieldtrial.web import console
from fieldtrial.web.security import SecurityMiddleware

STATIC = Path(__file__).parent / "static"

# fieldtrial never sends telemetry. FastAPI's OpenTelemetry hooks stay off even when the
# environment configures an exporter.
TELEMETRY_OFF: dict[str, Any] = {
    "tracing": False,
    "metrics": False,
    "logs": False,
    "operation_spans": False,
    "auto_configure": False,
}


def _status_for(exc: ServiceError) -> int:
    if isinstance(exc, UnknownStudyError):
        return 404
    if isinstance(exc, ConcurrencyError):
        return 409
    return 400


def create_app(
    root: str | Path,
    *,
    lan_token: str | None = None,
    allowed_hosts: Iterable[str] | None = None,
) -> FastAPI:
    """Build the app for the study (or folder of studies) at ``root``."""
    registry = StudyRegistry(root)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        registry.close()

    app = FastAPI(
        title="fieldtrial",
        version=API_VERSION,
        summary="Run robot-policy evaluation studies: schedule, record, analyze.",
        docs_url=None,  # the Swagger and ReDoc pages load scripts from a CDN
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
        telemetry=TELEMETRY_OFF,  # type: ignore[arg-type]
    )
    app.state.registry = registry
    app.state.lan = lan_token is not None
    app.state.runners = console.Runners()

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> Response:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": str(exc)}, status_code=_status_for(exc))
        return console.error_page(request, str(exc), _status_for(exc))

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> Response:
        if request.url.path.startswith("/api/"):
            return await request_validation_exception_handler(request, exc)
        fields = sorted({str(e["loc"][-1]).replace("_", " ") for e in exc.errors()})
        return console.error_page(request, f"Please fill in: {', '.join(fields)}.", 422)

    app.include_router(api_router)
    app.include_router(console.router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.add_middleware(SecurityMiddleware, lan_token=lan_token, allowed_hosts=allowed_hosts)
    return app
