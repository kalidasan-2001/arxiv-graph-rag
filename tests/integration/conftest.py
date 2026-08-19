"""Fixtures for tests that need a real PostgreSQL instance.

These tests are deliberately NOT run against SQLite: this stage relies on
PostgreSQL-specific behavior (JSONB columns, real unique-constraint
conflict handling) that SQLite cannot validate faithfully.

If no reachable database is configured, every test using the `db_session`
fixture is skipped (not failed) -- `pytest` alone must always succeed, per
the project's Prompt 0 contract, without requiring Docker/Postgres.

Point `TEST_DATABASE_URL` at a disposable database to run these for real,
e.g.:

    TEST_DATABASE_URL=postgresql+psycopg://arxiv:arxiv@localhost:5432/arxiv_graph_rag \
        pytest tests/integration

Falls back to the normal `DATABASE_URL` setting if `TEST_DATABASE_URL` is
not set.
"""

import os
import uuid
from collections.abc import Iterator

import pytest
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.storage.postgres.base import Base

# Import every ORM model module so `Base.metadata.create_all` (used only
# here, in isolated tests -- never as the production schema strategy) sees
# the full schema.
from app.storage.postgres.models import (  # noqa: F401
    IngestionJobRecord,
    IngestionStepRecord,
    PaperRecord,
    PaperVersionRecord,
)


def _resolve_test_database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL") or get_settings().DATABASE_URL


@pytest.fixture(scope="session")
def pg_engine() -> Iterator[Engine]:
    url = _resolve_test_database_url()
    if not url:
        pytest.skip("no DATABASE_URL/TEST_DATABASE_URL configured for PostgreSQL integration tests")

    engine = create_engine(url, pool_pre_ping=True, future=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError as exc:
        engine.dispose()
        pytest.skip(f"PostgreSQL not reachable at configured URL: {exc}")

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(pg_engine: Engine) -> Iterator[Session]:
    """A `Session` bound to a connection-scoped transaction that is always
    rolled back at teardown.

    Repository code under test only ever calls `session.flush()`, never
    `commit()` (only `SessionFactory` commits), so a plain rollback here is
    sufficient to leave the database untouched between tests -- no
    SAVEPOINT gymnastics required.
    """

    connection = pg_engine.connect()
    transaction = connection.begin()
    session_class = sessionmaker(bind=connection, expire_on_commit=False, future=True)
    session = session_class()

    yield session

    session.close()
    # A test that triggers a real IntegrityError (e.g. a unique-constraint
    # violation) already causes the DBAPI to deassociate the transaction;
    # only roll back if it's still active to avoid a spurious SAWarning.
    if transaction.is_active:
        transaction.rollback()
    connection.close()


# --- Qdrant --------------------------------------------------------------
#
# Same "skip, don't fail, when unreachable" policy as `pg_engine` above.
# Point `TEST_QDRANT_URL` at a disposable Qdrant instance to run these for
# real, e.g.:
#
#     TEST_QDRANT_URL=http://localhost:6333 pytest tests/integration


def _resolve_test_qdrant_url() -> str | None:
    return os.environ.get("TEST_QDRANT_URL") or get_settings().QDRANT_URL


@pytest.fixture(scope="session")
def qdrant_client() -> Iterator[QdrantClient]:
    url = _resolve_test_qdrant_url()
    if not url:
        pytest.skip("no QDRANT_URL/TEST_QDRANT_URL configured for Qdrant integration tests")

    client = QdrantClient(url=url, timeout=10)
    try:
        client.get_collections()
    except ApiException as exc:
        client.close()
        pytest.skip(f"Qdrant not reachable at configured URL: {exc}")

    yield client
    client.close()


@pytest.fixture
def qdrant_collection_name(qdrant_client: QdrantClient) -> Iterator[str]:
    """A uniquely-named, disposable collection for one test.

    Never reuses the application's real `QDRANT_COLLECTION` -- tests get
    their own throwaway collection so they can never interfere with (or be
    polluted by) real indexed data, and it's always cleaned up afterward
    regardless of whether the test itself created it.
    """

    name = f"test_collection_{uuid.uuid4().hex[:12]}"
    yield name
    if qdrant_client.collection_exists(name):
        qdrant_client.delete_collection(name)


# --- Neo4j --------------------------------------------------------------
#
# Same "skip, don't fail, when unreachable" policy as `pg_engine`/
# `qdrant_client` above. Point `TEST_NEO4J_URI`/`TEST_NEO4J_USERNAME`/
# `TEST_NEO4J_PASSWORD` at a *dedicated, disposable* Neo4j instance to run
# these for real -- e.g.:
#
#     TEST_NEO4J_URI=bolt://localhost:17687 \
#     TEST_NEO4J_USERNAME=neo4j TEST_NEO4J_PASSWORD=testpassword \
#     pytest tests/integration
#
# Falls back to `Settings.NEO4J_*` (mirrors the `DATABASE_URL`/`QDRANT_URL`
# fallback above) if the `TEST_NEO4J_*` variables aren't set -- this is a
# plain connection URI/credential, read only to construct a driver
# connection, never logged or asserted back out (unlike the LLM-API-key
# leak this project's `tests/llm/` module had to fix -- see that module's
# `_settings()` helper docstring for that incident).


def _resolve_test_neo4j_connection() -> tuple[str, str, str] | None:
    settings = get_settings()
    uri = os.environ.get("TEST_NEO4J_URI") or settings.NEO4J_URI
    username = os.environ.get("TEST_NEO4J_USERNAME") or settings.NEO4J_USERNAME
    password = os.environ.get("TEST_NEO4J_PASSWORD") or settings.NEO4J_PASSWORD
    if not uri or not username or not password:
        return None
    return uri, username, password


@pytest.fixture(scope="session")
def neo4j_driver() -> Iterator[Driver]:
    resolved = _resolve_test_neo4j_connection()
    if resolved is None:
        pytest.skip(
            "no NEO4J_URI/TEST_NEO4J_URI (+ credentials) configured for Neo4j integration tests"
        )
    uri, username, password = resolved

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
    except (Neo4jError, DriverError) as exc:
        driver.close()
        pytest.skip(f"Neo4j not reachable at configured URI: {exc}")

    yield driver
    driver.close()


@pytest.fixture
def neo4j_driver_clean(neo4j_driver: Driver) -> Iterator[Driver]:
    """`neo4j_driver`, wiped clean before and after the wrapped test.

    Community Edition (this project's target, CLAUDE.md #44) supports
    only the single default database -- there is no disposable per-test
    *database* the way `qdrant_collection_name` gets a disposable
    *collection* (prompt #73's "dedicated test database if the edition
    supports it cleanly, else..."). A full wipe is the simplest correct
    isolation strategy for a instance that is itself dedicated to tests --
    never point `TEST_NEO4J_URI` at a developer's real graph.
    """

    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield neo4j_driver
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def neo4j_repository(neo4j_driver_clean: Driver):
    from app.graph.neo4j_repository import Neo4jGraphRepository

    settings = get_settings()
    database = os.environ.get("TEST_NEO4J_DATABASE") or settings.NEO4J_DATABASE
    return Neo4jGraphRepository(neo4j_driver_clean, database)
