"""Deterministic fingerprints for the extraction stage (prompt #22/#23).

Mirrors `app.ingestion.chunking.fingerprint`/`app.embeddings.fingerprint`
exactly: `EXTRACTION_VERSION` alone is a human-maintained label nothing
forces a developer to bump when the LLM model, prompt template, or
structured schema changes -- these fingerprints capture every input that
actually determines extraction output.
"""

import hashlib
import json


def build_extraction_config_fingerprint(
    *,
    extraction_version: str,
    llm_provider: str,
    llm_model: str,
    llm_provider_version: str | None,
    prompt_version: str,
    schema_version: str,
    temperature: float,
) -> str:
    """SHA-256 hex digest of the canonical JSON of every field that
    materially affects extraction output."""

    canonical = {
        "extraction_version": extraction_version,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_provider_version": llm_provider_version,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "temperature": temperature,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def build_graph_extraction_generation_fingerprint(
    *, chunk_artifact_checksum: str, extraction_config_fingerprint: str
) -> str:
    """SHA-256 hex digest of the canonical JSON of the two inputs that
    together determine an extraction generation's identity: "exact chunks
    + exact extraction behavior." Mirrors
    `app.ingestion.vector_indexing.fingerprint.build_vector_generation_fingerprint`."""

    canonical = {
        "chunk_artifact_checksum": chunk_artifact_checksum,
        "extraction_config_fingerprint": extraction_config_fingerprint,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
