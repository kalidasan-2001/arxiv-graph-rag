#!/usr/bin/env python
"""Manual development script: run a real semantic vector search (prompt #68).

Requires a reachable Qdrant (QDRANT_URL) with at least one paper already
indexed. Does not call an LLM -- prints ranked chunks only.

Usage:
    python scripts/vector_search.py "graph neural network attack"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.embeddings.provider import get_embedding_provider  # noqa: E402
from app.retrieval.vector_search import VectorSearchService  # noqa: E402
from app.storage.qdrant.qdrant_repository import get_vector_repository  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python scripts/vector_search.py "<query text>"', file=sys.stderr)
        raise SystemExit(1)

    query = sys.argv[1]
    settings = get_settings()

    service = VectorSearchService(
        get_embedding_provider(),
        get_vector_repository(),
        default_top_k=settings.VECTOR_SEARCH_DEFAULT_TOP_K,
        max_top_k=settings.VECTOR_SEARCH_MAX_TOP_K,
    )
    results = service.search(query)

    if not results:
        print("No results.")
        return

    for i, hit in enumerate(results, start=1):
        print(f"{i}. score={hit.similarity_score:.4f}")
        print(f"   paper={hit.paper_id}")
        print(f"   section={hit.section_type} ({hit.section_title or 'untitled'})")
        print(f"   pages={hit.page_start}-{hit.page_end}")
        print(f"   chunk_id={hit.chunk_id}")
        print()


if __name__ == "__main__":
    main()
