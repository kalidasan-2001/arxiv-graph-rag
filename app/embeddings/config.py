"""The effective embedding configuration DTO (prompt #6).

Mirrors `app.ingestion.chunking.models.ChunkingConfig`: a small, serializable
snapshot of exactly what produced a set of vectors.
"""

from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    """Exactly what produced a generation of vectors -- the identity
    surface reuse/invalidation compares against (prompt #6/#7/#8)."""

    provider: str
    model: str
    dimension: int
    normalize: bool
    provider_version: str | None = None
    config_fingerprint: str
