#!/usr/bin/env python
"""Manual deterministic graph retrieval helper (Prompt 10).

Examples:
    python scripts/graph_search.py paper-datasets paper:arxiv:2401.12345
    python scripts/graph_search.py papers-for-dataset entity:dataset:abc123
    python scripts/graph_search.py shared-datasets paper:arxiv:2401.12345
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.graph.client import get_neo4j_driver  # noqa: E402
from app.graph.neo4j_repository import Neo4jGraphRepository  # noqa: E402
from app.retrieval.graph_search import GraphRetrievalService, GraphSearchOperation  # noqa: E402

_COMMAND_TO_OPERATION = {
    "paper-methods": GraphSearchOperation.PAPER_METHODS,
    "paper-datasets": GraphSearchOperation.PAPER_DATASETS,
    "paper-tasks": GraphSearchOperation.PAPER_TASKS,
    "paper-authors": GraphSearchOperation.PAPER_AUTHORS,
    "paper-citations": GraphSearchOperation.PAPER_CITATIONS,
    "paper-cited-by": GraphSearchOperation.PAPER_CITED_BY,
    "papers-for-method": GraphSearchOperation.PAPERS_FOR_METHOD,
    "papers-for-dataset": GraphSearchOperation.PAPERS_FOR_DATASET,
    "papers-for-task": GraphSearchOperation.PAPERS_FOR_TASK,
    "shared-datasets": GraphSearchOperation.SHARED_DATASETS,
    "shared-methods": GraphSearchOperation.SHARED_METHODS,
    "datasets-from-citing-papers": GraphSearchOperation.DATASETS_FROM_CITING_PAPERS,
    "methods-for-dataset": GraphSearchOperation.METHODS_FOR_DATASET,
    "citation-neighborhood": GraphSearchOperation.CITATION_NEIGHBORHOOD,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Neo4j graph retrieval.")
    parser.add_argument("operation", choices=sorted(_COMMAND_TO_OPERATION))
    parser.add_argument("entity_id", help="Stable start entity id, e.g. paper:arxiv:2401.12345")
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    driver = get_neo4j_driver(settings)
    try:
        service = GraphRetrievalService(
            Neo4jGraphRepository(driver, settings.NEO4J_DATABASE),
            max_depth=settings.GRAPH_MAX_DEPTH,
            default_limit=settings.GRAPH_DEFAULT_LIMIT,
            max_limit=settings.GRAPH_MAX_LIMIT,
        )
        result = service.search(
            operation=_COMMAND_TO_OPERATION[args.operation],
            entity_id=args.entity_id,
            depth=args.depth,
            limit=args.limit,
        )
    finally:
        driver.close()

    print(f"operation: {result.operation.value}")
    print(f"start: {result.start_entity.entity_type}:{result.start_entity.canonical_name}")
    print(f"results: {len(result.results)}")
    print()

    for index, item in enumerate(result.results, start=1):
        print(f"{index}. {item.summary}")
        print(f"   evidence_id: {item.evidence_id}")
        print(f"   path_confidence: {item.path_confidence}")
        print("   entities:")
        for node in item.path.nodes:
            print(f"     - {node.entity_type} {node.entity_id} {node.canonical_name}")
        print("   relationships:")
        for relationship in item.path.relationships:
            chunks = [relationship.source_chunk_id, *relationship.supporting_chunk_ids]
            chunks = [chunk for chunk in chunks if chunk]
            print(
                "     - "
                f"{relationship.relationship_type} {relationship.relationship_id} "
                f"confidence={relationship.confidence} provenance={relationship.provenance_type} "
                f"chunks={chunks}"
            )
        print()


if __name__ == "__main__":
    main()
