"""The parser abstraction the rest of the application depends on.

Application code (`PaperParsingService`, tests) depends only on this
`Protocol` and `ParsedPaperDocument` -- never on a vendor PDF library type
-- so the concrete implementation (`pymupdf_parser.PyMuPDFParser` today)
can be replaced by a better scientific parser later without rewriting the
ingestion pipeline (prompt #3).
"""

from pathlib import Path
from typing import Protocol

from app.ingestion.parsing.models import ParsedPaperDocument


class ScientificPaperParser(Protocol):
    """A deterministic, local, page-aware PDF-to-structured-document parser."""

    @property
    def name(self) -> str:
        """Short identifier, e.g. `"pymupdf"`. Persisted as `parser_name`."""
        ...

    @property
    def version(self) -> str:
        """Implementation version. Persisted as `parser_version`.

        Should change whenever this parser's extraction/section-recovery
        behavior changes materially -- not just the underlying library
        version -- so a changed `version` reliably signals "this document
        should be reparsed" (prompt #9, #29).
        """
        ...

    def parse(
        self, artifact_path: Path, *, paper_id: str, paper_version_id: str
    ) -> ParsedPaperDocument:
        """Extract text and recover sections from a local PDF file.

        Raises `UnsupportedPdfError` if the file can't be opened at all
        (corrupt, encrypted, not a PDF), or `PdfParseError` for other
        extraction failures (e.g. no extractable text at all).
        """
        ...
