#!/usr/bin/env python
"""Manual smoke-test script: search arXiv, then really download and store one PDF.

NOT part of the automated test suite -- requires real network access AND a
reachable PostgreSQL (DATABASE_URL). Writes into PAPER_STORAGE_PATH for
real.

Usage:
    python scripts/ingest_arxiv_pdf.py 2401.12345
    python scripts/ingest_arxiv_pdf.py "graph rag"   # searches, ingests the first hit
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.discovery.arxiv_client import ArxivClient  # noqa: E402
from app.ingestion.discovery.models import PaperSearchQuery  # noqa: E402
from app.ingestion.discovery.normalization import (  # noqa: E402
    arxiv_result_to_paper,
    arxiv_result_to_paper_version,
)
from app.ingestion.download.client import PdfDownloadClient  # noqa: E402
from app.ingestion.download.service import PdfAcquisitionService  # noqa: E402
from app.ingestion.download.storage import PaperStorage  # noqa: E402
from app.storage.postgres.repositories.ingestion import IngestionRepository  # noqa: E402
from app.storage.postgres.repositories.papers import PaperRepository  # noqa: E402
from app.storage.postgres.session import SessionFactory  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/ingest_arxiv_pdf.py <arxiv-id-or-query>", file=sys.stderr)
        raise SystemExit(1)

    settings = get_settings()
    session_factory = SessionFactory(settings)
    query_text = " ".join(sys.argv[1:])

    with ArxivClient(settings) as arxiv_client:
        results = arxiv_client.search(PaperSearchQuery(query=query_text, max_results=1))
    if not results:
        print("No arXiv results.")
        return

    paper = arxiv_result_to_paper(results[0])
    print(f"Discovered: {paper.title} ({paper.source_id})")

    with session_factory() as session:
        repo = PaperRepository(session)
        paper = repo.upsert_paper(paper)
        version = arxiv_result_to_paper_version(results[0], paper_id=paper.paper_id)
        if version is not None:
            repo.get_or_create_paper_version(version)

    with session_factory() as session, PdfDownloadClient(settings) as pdf_client:
        service = PdfAcquisitionService(
            settings,
            pdf_client,
            PaperStorage(settings),
            PaperRepository(session),
            IngestionRepository(session),
        )
        result = service.ingest(paper.paper_id)

    print(f"ingestion_job_id: {result.job.ingestion_job_id}")
    print(f"status: {result.job.status.value}")
    print(f"artifact_reused: {result.artifact_reused}")


if __name__ == "__main__":
    main()
