"""Qdrant client construction.

Mirrors `app.ingestion.download.client.get_pdf_download_client`: a plain
factory (not process-wide cached) reading connection settings from
`Settings`, never a hard-coded host (CLAUDE.md #36/prompt #10).
"""

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.core.exceptions import ConfigurationError


def get_qdrant_client(settings: Settings) -> QdrantClient:
    if not settings.QDRANT_URL:
        raise ConfigurationError("QDRANT_URL is not configured")

    # An empty string in `.env` (the local-Docker-with-no-auth default,
    # prompt #10) is not the same as "no API key" to qdrant-client, which
    # warns whenever `api_key` is not `None` on a non-https URL -- normalize
    # blank to `None` so that warning only fires when a key is genuinely set.
    api_key = settings.QDRANT_API_KEY or None

    # Docker Compose pins the Qdrant image to match the installed
    # `qdrant-client` version (see docker-compose.yml) specifically so the
    # client's own compatibility check (left at its default here) has
    # nothing to warn about in the documented setup.
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=api_key,
        timeout=int(settings.QDRANT_TIMEOUT_SECONDS),
    )
