#!/usr/bin/env python
"""Manual development script: report a paper's vector-index reconciliation
state (prompt #69) -- useful for debugging idempotency/invalidation.

Requires the paper to already have a chunk artifact and a reachable
PostgreSQL (DATABASE_URL) + Qdrant (QDRANT_URL). Does not index anything
itself.

Usage:
    python scripts/inspect_vector_index.py paper:arxiv:2401.12345
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.embeddings.provider import get_embedding_provider  # noqa: E402
from app.ingestion.chunking.storage import ChunkArtifactStorage  # noqa: E402
from app.ingestion.checksums import sha256_file  # noqa: E402
from app.ingestion.paper_resolution import resolve_paper, resolve_version  # noqa: E402
from app.ingestion.vector_indexing.fingerprint import build_vector_generation_fingerprint  # noqa: E402
from app.storage.postgres.repositories.papers import PaperRepository  # noqa: E402
from app.storage.postgres.session import SessionFactory  # noqa: E402
from app.storage.qdrant.qdrant_repository import get_vector_repository  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/inspect_vector_index.py <paper_id>", file=sys.stderr)
        raise SystemExit(1)

    paper_id = sys.argv[1]
    settings = get_settings()
    session_factory = SessionFactory(settings)

    with session_factory() as session:
        papers = PaperRepository(session)
        paper = resolve_paper(papers, paper_id)
        version = resolve_version(papers, paper, None)

    if not version.chunked_artifact_path:
        print(f"{paper_id} has not been chunked yet.", file=sys.stderr)
        raise SystemExit(1)

    chunk_storage = ChunkArtifactStorage(settings)
    document = chunk_storage.try_read(
        source=paper.source, source_id=paper.source_id, version=version.version
    )
    if document is None:
        print(f"chunk artifact for {paper_id} is missing or corrupt.", file=sys.stderr)
        raise SystemExit(1)
    expected_chunks = len(document.chunks)
    chunk_checksum = sha256_file(
        chunk_storage.get_path(source=paper.source, source_id=paper.source_id, version=version.version)
    )

    provider = get_embedding_provider()
    embedding_config_fingerprint = provider.config_fingerprint
    current_generation_fingerprint = build_vector_generation_fingerprint(
        chunk_artifact_checksum=chunk_checksum,
        embedding_config_fingerprint=embedding_config_fingerprint,
    )

    vector_repo = get_vector_repository()
    current_count = vector_repo.count_for_paper_version(
        version.paper_version_id, generation_fingerprint=current_generation_fingerprint
    )
    total_count = vector_repo.count_for_paper_version(version.paper_version_id)

    print(f"Paper: {paper.title}")
    print(f"Version: {version.version}")
    print(f"Expected chunks: {expected_chunks}")
    print(f"Embedding model: {provider.model_name} (provider={provider.provider_name})")
    print(f"Current generation fingerprint: {current_generation_fingerprint[:16]}...")
    print(f"Points in Qdrant for this paper version (any generation): {total_count}")
    print(f"Points matching the current generation: {current_count}")

    if current_count == expected_chunks:
        status = "VALID (up to date)"
    elif current_count == 0:
        status = "MISSING (never indexed, or fully stale)"
    else:
        status = f"PARTIAL ({current_count}/{expected_chunks} current-generation points present)"
    print(f"Reconciliation status: {status}")

    print()
    print("PostgreSQL-recorded metadata:")
    print(f"  vector_count: {version.vector_count}")
    print(f"  embedding_provider: {version.embedding_provider}")
    print(f"  embedding_model: {version.embedding_model}")
    print(f"  vector_generation_fingerprint: {version.vector_generation_fingerprint}")
    print(f"  vector_indexed_at: {version.vector_indexed_at}")
    if version.vector_generation_fingerprint != current_generation_fingerprint:
        print("  NOTE: PostgreSQL's recorded fingerprint does not match the current one --")
        print("        this is expected before the next explicit /vector-index call reconciles it.")


if __name__ == "__main__":
    main()
