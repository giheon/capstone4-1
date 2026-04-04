from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes.solve import router as solve_router
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger
from app.domain.concepts.catalog import ConceptCatalog
from app.domain.similar.service import PlaceholderSimilarProblemsService
from app.llm.client import OpenAIResponsesClient
from app.schemas.response import ErrorResponse
from app.workflows.solve_graph import build_solve_graph


def create_app(
    settings: Settings | None = None,
    llm_client: object | None = None,
    similar_service: PlaceholderSimilarProblemsService | None = None,
    catalog: ConceptCatalog | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    configure_logging(runtime_settings.log_level)
    logger = get_logger("app")
    project_root = Path(__file__).resolve().parent.parent

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime_catalog = catalog or ConceptCatalog.from_project_root(project_root)
        runtime_settings.require_openai_api_key()
        runtime_llm_client = llm_client or OpenAIResponsesClient(runtime_settings)
        runtime_similar_service = similar_service or PlaceholderSimilarProblemsService()

        app.state.runtime = SimpleNamespace(
            settings=runtime_settings,
            catalog=runtime_catalog,
            llm_client=runtime_llm_client,
            similar_service=runtime_similar_service,
            logger=logger,
        )
        app.state.solve_graph = build_solve_graph(
            settings=runtime_settings,
            catalog=runtime_catalog,
            llm_client=runtime_llm_client,
            similar_service=runtime_similar_service,
            logger=logger,
        )
        yield

    app = FastAPI(
        title="Suneung Math Solver Backend",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(solve_router)

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                f"{request.method} {request.url.path} completed in {duration_ms}ms",
                extra={"request_id": request_id},
            )
            raise
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            f"{request.method} {request.url.path} completed in {duration_ms}ms",
            extra={"request_id": request_id},
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        payload = ErrorResponse(
            request_id=getattr(request.state, "request_id", "unknown"),
            error_code=exc.error_code,
            message=exc.message,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.warning(
            f"Request validation failed: {exc.errors()}",
            extra={"request_id": getattr(request.state, "request_id", "unknown")},
        )
        payload = ErrorResponse(
            request_id=getattr(request.state, "request_id", "unknown"),
            error_code="INVALID_REQUEST_FORMAT",
            message="잘못된 요청 형식입니다.",
        )
        return JSONResponse(status_code=400, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled server error.",
            extra={"request_id": getattr(request.state, "request_id", "unknown")},
        )
        payload = ErrorResponse(
            request_id=getattr(request.state, "request_id", "unknown"),
            error_code="INTERNAL_SERVER_ERROR",
            message="서버 내부 오류가 발생했습니다.",
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    return app


app = create_app()
