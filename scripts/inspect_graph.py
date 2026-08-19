#!/usr/bin/env python
"""Manual development script: print a paper's directly-connected Neo4j
knowledge graph (prompt #76).

Requires the paper to already have been graph-indexed (via the API or
`GraphIndexingService`) and a reachable Neo4j (NEO4J_URI). Read-only --
never mutates the graph, and never dumps raw Neo4j driver internals.

Usage:
    python scripts/inspect_graph.py paper:arxiv:2401.12345
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.graph.client import get_neo4j_driver  # noqa: E402
from app.graph.neo4j_repository import Neo4jGraphRepository  # noqa: E402

_SECTION_BY_TYPE = {
    "author": "AUTHORS",
    "method": "METHODS",
    "dataset": "DATASETS",
    "task": "TASKS",
}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/inspect_graph.py <paper_id>", file=sys.stderr)
        raise SystemExit(1)

    paper_id = sys.argv[1]
    settings = get_settings()
    driver = get_neo4j_driver(settings)
    repository = Neo4jGraphRepository(driver, settings.NEO4J_DATABASE)

    try:
        paper, nodes, relationships = repository.get_paper_graph(paper_id)
    finally:
        driver.close()

    if paper is None:
        print(f"{paper_id} has not been graph-indexed yet.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Paper: {paper.properties.get('title', paper.canonical_name)}")
    print(f"Nodes: {len(nodes)}  Relationships: {len(relationships)}")
    print()

    nodes_by_id = {node.entity_id: node for node in nodes}
    grouped: dict[str, list] = {}
    cites: list = []
    for relationship in relationships:
        target = nodes_by_id.get(relationship.target_entity_id)
        if target is None:
            continue
        if relationship.relationship_type == "cites":
            cites.append(target)
            continue
        grouped.setdefault(target.entity_type, []).append(target)

    for entity_type, heading in _SECTION_BY_TYPE.items():
        entries = grouped.get(entity_type, [])
        if not entries:
            continue
        print(heading)
        for entry in entries:
            print(f"- {entry.canonical_name}")
        print()

    if cites:
        print("CITES")
        for target in cites:
            marker = " (placeholder)" if target.properties.get("is_placeholder") else ""
            print(f"- {target.properties.get('title', target.canonical_name)}{marker}")


if __name__ == "__main__":
    main()
