"""Deterministic fingerprint of the complete *effective* chunking
configuration (prompt 6.1).

Why this exists: `CHUNKING_VERSION` alone is a human-maintained label, not
a guarantee. Nothing forces a developer to bump it when `CHUNK_SIZE_TOKENS`,
`CHUNK_OVERLAP_TOKENS`, `MIN_CHUNK_TOKENS`, or the tokenizer implementation
changes -- and a stale, unbumped version means a materially different
chunking configuration can silently reuse an old, no-longer-representative
`chunks.json`. `build_chunk_config_fingerprint` instead derives a stable
identity from every input that actually affects chunk output, so reuse and
chunk-id collisions are governed by what the configuration *is*, not by
whether someone remembered to rename it.
"""

import hashlib
import json


def build_chunk_config_fingerprint(
    *,
    chunking_version: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    min_chunk_tokens: int,
    tokenizer_name: str,
    tokenizer_version: str | None,
) -> str:
    """SHA-256 hex digest of the canonical JSON serialization of every
    field that materially affects chunking output.

    Deterministic across processes and Python versions -- never Python's
    built-in `hash()`, which is randomized per-process for strings.
    `sort_keys=True` and fixed compact separators make the JSON
    serialization canonical regardless of the order these keyword
    arguments were supplied in, so two calls with the same *values* always
    produce the same fingerprint, independent of object identity.
    """

    canonical = {
        "chunking_version": chunking_version,
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "min_chunk_tokens": min_chunk_tokens,
        "tokenizer_name": tokenizer_name,
        "tokenizer_version": tokenizer_version,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
