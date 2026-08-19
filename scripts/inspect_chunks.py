#!/usr/bin/env python
"""Manual development script: print a summary of a chunked paper's structure.

Requires the paper to already have a persisted chunk result (i.e. it was
chunked via the API or `ChunkingService`) and a reachable PostgreSQL
(DATABASE_URL). Does not chunk anything itself, and does not dump full
chunk text by default.

Usage:
    python scripts/inspect_chunks.py paper:arxiv:2401.12345
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.chunking.storage import ChunkArtifactStorage  # noqa: E402
from app.ingestion.paper_resolution import resolve_paper, resolve_version  # noqa: E402
from app.storage.postgres.repositories.papers import PaperRepository  # noqa: E402
from app.storage.postgres.session import SessionFactory  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/inspect_chunks.py <paper_id>", file=sys.stderr)
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

    document = ChunkArtifactStorage(settings).try_read(
        source=paper.source, source_id=paper.source_id, version=version.version
    )
    if document is None:
        print(f"chunk artifact for {paper_id} is missing or corrupt.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Paper: {paper.title}")
    print(f"Version: {version.version}")
    print(f"Chunks: {document.diagnostics.chunk_count}")
    print(
        f"Tokens: min={document.diagnostics.min_tokens} "
        f"max={document.diagnostics.max_tokens} "
        f"avg={document.diagnostics.average_tokens:.1f} "
        f"median={document.diagnostics.median_tokens:.1f}"
    )
    print(
        f"Chunking: version={document.chunking.version} "
        f"size={document.chunking.chunk_size_tokens} "
        f"overlap={document.chunking.chunk_overlap_tokens} "
        f"tokenizer={document.chunking.tokenizer}"
    )
    # Truncated for readability -- the full fingerprint is what reuse and
    # chunk_id derivation actually compare (prompt 6.1).
    print(f"Config fingerprint: {document.chunking.config_fingerprint[:16]}...")
    if document.warnings:
        print(f"Warnings: {', '.join(w.value for w in document.warnings)}")
    print()

    by_section: dict[str, list] = defaultdict(list)
    for chunk in document.chunks:
        by_section[chunk.section_id].append(chunk)

    for section_id, chunks in by_section.items():
        first = chunks[0]
        title = first.metadata.get("section_title") or "(untitled)"
        print(f"[{first.section_type.value.upper()}] -- {title}")
        for chunk in sorted(chunks, key=lambda c: c.chunk_index):
            print(
                f"    chunk {chunk.chunk_index}: tokens={chunk.token_count} "
                f"pages={chunk.page_start}-{chunk.page_end} id={chunk.chunk_id}"
            )
        print()


if __name__ == "__main__":
    main()
