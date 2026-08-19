#!/usr/bin/env python
"""Manual development script: print a summary of a paper's graph
extraction artifact (prompt #62).

Requires the paper to already have a persisted extraction result (i.e. it
was extracted via the API or `ScientificKnowledgeExtractionService`) and a
reachable PostgreSQL (DATABASE_URL). Does not extract anything itself, and
does not dump the full artifact JSON by default.

Usage:
    python scripts/inspect_graph_extraction.py paper:arxiv:2401.12345
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.graph_extraction.storage import GraphExtractionArtifactStorage  # noqa: E402
from app.ingestion.paper_resolution import resolve_paper, resolve_version  # noqa: E402
from app.storage.postgres.repositories.papers import PaperRepository  # noqa: E402
from app.storage.postgres.session import SessionFactory  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/inspect_graph_extraction.py <paper_id>", file=sys.stderr)
        raise SystemExit(1)

    paper_id = sys.argv[1]
    settings = get_settings()
    session_factory = SessionFactory(settings)

    with session_factory() as session:
        papers = PaperRepository(session)
        paper = resolve_paper(papers, paper_id)
        version = resolve_version(papers, paper, None)

    if not version.graph_extraction_artifact_path:
        print(f"{paper_id} has not been graph-extracted yet.", file=sys.stderr)
        raise SystemExit(1)

    artifact = GraphExtractionArtifactStorage(settings).try_read(
        source=paper.source, source_id=paper.source_id, version=version.version
    )
    if artifact is None:
        print(f"graph extraction artifact for {paper_id} is missing or corrupt.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Paper: {paper.title}")
    print(f"Entities: {len(artifact.entities)}")
    print(f"Relationships: {len(artifact.relationships)}")
    print(f"Unresolved citations: {len(artifact.unresolved_citations)}")
    print(
        f"Extraction: version={artifact.extraction.version} "
        f"llm={artifact.extraction.llm_provider}/{artifact.extraction.llm_model} "
        f"prompt_version={artifact.extraction.prompt_version}"
    )
    if artifact.warnings:
        print(f"Warnings: {len(artifact.warnings)}")
    print()

    by_type: dict[str, list] = {}
    for entity in artifact.entities:
        by_type.setdefault(entity.entity_type.value.upper(), []).append(entity)

    for entity_type in ("METHOD", "DATASET", "TASK", "AUTHOR"):
        entities = by_type.get(entity_type, [])
        if not entities:
            continue
        print(entity_type)
        for entity in entities:
            print(f"- {entity.canonical_name}")
        print()

    print("RELATIONSHIPS")
    entity_names = {entity.entity_id: entity.canonical_name for entity in artifact.entities}
    for relationship in artifact.relationships:
        source_name = entity_names.get(relationship.source_entity_id, relationship.source_entity_id)
        target_name = entity_names.get(relationship.target_entity_id, relationship.target_entity_id)
        print(f"{source_name} -> {relationship.relationship_type.value} -> {target_name}")

    if artifact.unresolved_citations:
        print()
        print("UNRESOLVED CITATIONS")
        for citation in artifact.unresolved_citations:
            print(f"- {citation.raw_text[:80]}... ({citation.reason})")


if __name__ == "__main__":
    main()
