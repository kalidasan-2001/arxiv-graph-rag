#!/usr/bin/env python
"""Manual development script: print a summary of a parsed paper's structure.

Requires the paper to already have a persisted parse result (i.e. it was
parsed via the API or `PaperParsingService`) and a reachable PostgreSQL
(DATABASE_URL). Does not parse anything itself, and does not dump full
section bodies by default.

Usage:
    python scripts/inspect_parsed_paper.py paper:arxiv:2401.12345
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.paper_resolution import resolve_paper, resolve_version  # noqa: E402
from app.ingestion.parsing.storage import ParsedArtifactStorage  # noqa: E402
from app.storage.postgres.repositories.papers import PaperRepository  # noqa: E402
from app.storage.postgres.session import SessionFactory  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/inspect_parsed_paper.py <paper_id>", file=sys.stderr)
        raise SystemExit(1)

    paper_id = sys.argv[1]
    settings = get_settings()
    session_factory = SessionFactory(settings)

    with session_factory() as session:
        papers = PaperRepository(session)
        paper = resolve_paper(papers, paper_id)
        version = resolve_version(papers, paper, None)

    if not version.parsed_artifact_path:
        print(f"{paper_id} has not been parsed yet.", file=sys.stderr)
        raise SystemExit(1)

    document = ParsedArtifactStorage(settings).try_read(
        source=paper.source, source_id=paper.source_id, version=version.version
    )
    if document is None:
        print(f"parsed artifact for {paper_id} is missing or corrupt.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Paper: {paper.title}")
    print(f"Pages: {document.page_count}")
    print(f"Parser: {document.parser_name} {document.parser_version}")
    if document.warnings:
        print(f"Warnings: {', '.join(w.value for w in document.warnings)}")
    print()

    for i, section in enumerate(document.sections, start=1):
        title = section.title or "(untitled)"
        print(f"[{i}] {section.section_type.value.upper()} -- {title}")
        print(f"    pages: {section.page_start}-{section.page_end}")
        print(f"    chars: {len(section.text)}")
        print()


if __name__ == "__main__":
    main()
