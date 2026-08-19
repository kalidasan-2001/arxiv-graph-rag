#!/usr/bin/env python
"""Manual smoke-test script: search live arXiv and print results.

NOT part of the automated test suite -- requires real network access to
arXiv. Does not persist anything to PostgreSQL.

Usage:
    python scripts/search_arxiv.py "graph rag"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.discovery.arxiv_client import ArxivClient  # noqa: E402
from app.ingestion.discovery.models import PaperSearchQuery  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/search_arxiv.py <query>", file=sys.stderr)
        raise SystemExit(1)

    query_text = " ".join(sys.argv[1:])
    settings = get_settings()
    query = PaperSearchQuery(query=query_text, max_results=5)

    with ArxivClient(settings) as client:
        results = client.search(query)

    if not results:
        print("No results.")
        return

    for result in results:
        print(result.title)
        print(f"  arxiv id: {result.source_id} ({result.version or 'no version'})")
        print(f"  published: {result.published_at}")
        print()


if __name__ == "__main__":
    main()
