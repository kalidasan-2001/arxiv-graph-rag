"""Neo4j driver construction.

Mirrors `app.storage.qdrant.client.get_qdrant_client`: a plain factory
(not process-wide cached) reading connection settings from `Settings`,
never a hard-coded host/credential (CLAUDE.md #29/#36).
"""

from neo4j import Driver, GraphDatabase

from app.core.config import Settings
from app.core.exceptions import ConfigurationError


def get_neo4j_driver(settings: Settings) -> Driver:
    if not settings.NEO4J_URI:
        raise ConfigurationError("NEO4J_URI is not configured")
    if not settings.NEO4J_USERNAME or not settings.NEO4J_PASSWORD:
        raise ConfigurationError("NEO4J_USERNAME/NEO4J_PASSWORD is not configured")

    # Connectivity is not verified here (mirrors `get_qdrant_client`) --
    # construction never blocks on the network; the first real operation
    # surfaces `GraphStoreUnavailableError` if Neo4j is unreachable.
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        connection_timeout=settings.NEO4J_TIMEOUT_SECONDS,
    )
