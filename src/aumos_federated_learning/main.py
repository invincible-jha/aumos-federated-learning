"""FastAPI application factory for aumos-federated-learning."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aumos_federated_learning.api.router import router
from aumos_federated_learning.settings import get_settings


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown hooks."""
    settings = get_settings()

    # Startup
    application.state.settings = settings

    yield

    # Shutdown: nothing to teardown in scaffolding phase


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title="AumOS Federated Learning",
        description="Enterprise federated learning without data sharing",
        version=settings.service_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)

    @application.get("/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        """Liveness probe — returns 200 if the process is alive."""
        return {"status": "alive"}

    @application.get("/ready", tags=["health"])
    async def readiness() -> dict[str, str]:
        """Readiness probe — returns 200 if the service is ready to accept traffic."""
        return {"status": "ready"}

    return application


app = create_app()
