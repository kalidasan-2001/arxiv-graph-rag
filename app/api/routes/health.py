"""Health check endpoints."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from neo4j.exceptions import DriverError, Neo4jError
from pydantic import BaseModel
from qdrant_client.http.exceptions import ApiException
from sqlalchemy import text

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.graph.client import get_neo4j_driver
from app.storage.postgres.session import get_session_factory
from app.storage.qdrant.client import get_qdrant_client

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response body for the liveness health check."""

    status: str
    service: str


class DatabaseHealthResponse(BaseModel):
    """Response body for the database readiness check."""

    status: str
    database: str


class QdrantHealthResponse(BaseModel):
    """Response body for the Qdrant readiness check."""

    status: str
    qdrant: str


class Neo4jHealthResponse(BaseModel):
    """Response body for the Neo4j readiness check."""

    status: str
    neo4j: str


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report basic service liveness. Never depends on PostgreSQL."""

    return HealthResponse(status="healthy", service="arxiv-graph-rag-api")


@router.get("/health/db", response_model=None)
def get_database_health() -> DatabaseHealthResponse | JSONResponse:
    """Report whether PostgreSQL is currently reachable (readiness, not liveness).

    Kept deliberately separate from `/health`: the API process is alive
    even when the database is unreachable, so liveness must not fail
    solely because PostgreSQL is down.
    """

    try:
        session_factory = get_session_factory()
    except ConfigurationError:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "database": "not_configured"},
        )

    try:
        with session_factory() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "database": "unreachable"},
        )

    return DatabaseHealthResponse(status="healthy", database="connected")


@router.get("/health/qdrant", response_model=None)
def get_qdrant_health(settings: Settings = Depends(get_settings)) -> QdrantHealthResponse | JSONResponse:
    """Report whether Qdrant is currently reachable (readiness, not liveness).

    Same pattern as `/health/db`: process liveness (`/health`) must never
    fail merely because Qdrant is temporarily unavailable (prompt #49).
    """

    try:
        client = get_qdrant_client(settings)
        client.get_collections()
    except (ConfigurationError, ApiException):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "qdrant": "unreachable"},
        )

    return QdrantHealthResponse(status="healthy", qdrant="connected")


@router.get("/health/neo4j", response_model=None)
def get_neo4j_health(settings: Settings = Depends(get_settings)) -> Neo4jHealthResponse | JSONResponse:
    """Report whether Neo4j is currently reachable (readiness, not liveness).

    Same pattern as `/health/db`/`/health/qdrant`: process liveness
    (`/health`) must never fail merely because Neo4j is temporarily
    unavailable (CLAUDE.md #43).
    """

    try:
        driver = get_neo4j_driver(settings)
        driver.verify_connectivity()
        driver.close()
    except (ConfigurationError, Neo4jError, DriverError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "neo4j": "unreachable"},
        )

    return Neo4jHealthResponse(status="healthy", neo4j="connected")
