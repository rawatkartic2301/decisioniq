"""FastAPI application factory and middleware configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from decisioniq.api.middleware import (
    RateLimitMiddleware,
    RequestIdMiddleware,
    StructuredLoggingMiddleware,
)
from decisioniq.api.routes import router


def create_app() -> FastAPI:
    """Creates and configures the DecisionIQ production FastAPI instance."""
    app = FastAPI(
        title="DecisionIQ Enterprise API",
        description=(
            "Agentic RAG Assistant combining unstructured HR/payroll policies, "
            "relational SQLite business records, and unsupervised ML anomaly detection."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Custom middlewares (applied in reverse order of addition)
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=120, window_seconds=60)
    app.add_middleware(RequestIdMiddleware)

    # 3. Mount routes
    app.include_router(router)

    @app.get("/", tags=["Root"])
    def root():
        return {
            "name": "DecisionIQ API",
            "version": "1.0.0",
            "status": "online",
            "docs_url": "/docs",
        }

    return app


app = create_app()
