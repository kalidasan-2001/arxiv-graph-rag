"""Deterministic fingerprint of the complete effective embedding
configuration (prompt #7).

Mirrors `app.ingestion.chunking.fingerprint.build_chunk_config_fingerprint`
exactly, for the same reason: `EMBEDDING_MODEL` alone is not sufficient
proof that existing vectors are still valid. A dimension change, a
normalization-behavior change, or a provider-implementation upgrade can all
make existing vectors incompatible or semantically different without the
model name itself changing (prompt #8) -- this fingerprint captures every
input that actually determines vector output, not just its label.
"""

import hashlib
import json


def build_embedding_config_fingerprint(
    *,
    provider: str,
    model: str,
    dimension: int,
    normalize: bool,
    provider_version: str | None,
) -> str:
    """SHA-256 hex digest of the canonical JSON serialization of every
    field that materially affects embedding output.

    Deterministic across processes -- never Python's built-in `hash()`.
    `sort_keys=True` and fixed compact separators make the JSON
    serialization canonical regardless of keyword-argument order.
    """

    canonical = {
        "provider": provider,
        "model": model,
        "dimension": dimension,
        "normalize": normalize,
        "provider_version": provider_version,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
