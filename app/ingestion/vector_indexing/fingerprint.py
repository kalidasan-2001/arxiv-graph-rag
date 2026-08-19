"""Vector generation identity (prompt #25).

A "vector generation" is the complete set of a paper version's current
vectors. It's uniquely determined by *what* was embedded (the chunk
artifact) and *how* (the embedding configuration) -- this fingerprint lets
`VectorIndexingService` answer "do the points currently in Qdrant belong to
exactly this generation?" without re-embedding anything.
"""

import hashlib
import json


def build_vector_generation_fingerprint(
    *, chunk_artifact_checksum: str, embedding_config_fingerprint: str
) -> str:
    """SHA-256 hex digest of the canonical JSON of the two inputs that
    together determine a vector generation's identity.

    Canonical (sorted-key) JSON serialization, not naive string
    concatenation (prompt #25) -- avoids any ambiguity from where one
    field's value ends and the next begins.
    """

    canonical = {
        "chunk_artifact_checksum": chunk_artifact_checksum,
        "embedding_config_fingerprint": embedding_config_fingerprint,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
