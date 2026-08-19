"""PyMuPDF-based `ScientificPaperParser` implementation.

PyMuPDF (`pymupdf`, formerly `fitz`) was chosen because it gives fast,
reliable page-by-page plain-text extraction with a natural page-provenance
model (one call per page), has no external runtime dependencies or
separate service to run (unlike GROBID), and is mature/widely used enough
for deterministic, reproducible output -- exactly the properties this
stage needs (prompt #4).

Known limitations:
* Two-column layout reading order is not reconstructed -- only detected
  and flagged as a warning (prompt #12: no layout/CV engine in V1).
* Scanned/image-only PDFs yield little or no text; flagged via
  `LOW_TEXT_DENSITY`, never OCR'd (prompt #38).
* This is the only parser in V1 -- if it can't handle a paper, parsing
  fails with a clear reason rather than falling back to a second engine
  (prompt #37).

This is the *only* module in the codebase that imports `pymupdf` --
nothing else may depend on it directly (prompt #3).
"""

from pathlib import Path

import pymupdf

from app.core.exceptions import PdfParseError, UnsupportedPdfError
from app.ingestion.parsing import normalization, section_recovery
from app.ingestion.parsing.models import ParsedPage, ParsedPaperDocument, ParseWarning

_PARSER_NAME = "pymupdf"
# Our own extraction/section-recovery adapter logic version -- independent
# of the pymupdf library version, so a change to *this* module's behavior
# (not just a library upgrade) still invalidates cached parses (prompt #9).
_ADAPTER_VERSION = "1"

# A block is only considered a real two-column signal if it has some
# meaningful width -- filters out thin decorative/rule-line blocks.
_MIN_BLOCK_WIDTH = 20


class PyMuPDFParser:
    """`ScientificPaperParser` implementation backed by PyMuPDF."""

    @property
    def name(self) -> str:
        return _PARSER_NAME

    @property
    def version(self) -> str:
        return f"{pymupdf.__version__}+adapter{_ADAPTER_VERSION}"

    def parse(
        self, artifact_path: Path, *, paper_id: str, paper_version_id: str
    ) -> ParsedPaperDocument:
        document = _open_pdf(artifact_path)
        try:
            if document.is_encrypted:
                raise UnsupportedPdfError(f"PDF is password-protected/encrypted: {artifact_path}")

            raw_pages: list[ParsedPage] = []
            two_column_pages = 0
            for page_index in range(document.page_count):
                page = document[page_index]
                raw_pages.append(ParsedPage(page_number=page_index + 1, text=page.get_text("text")))
                if _looks_two_column(page):
                    two_column_pages += 1
        except UnsupportedPdfError:
            raise
        except Exception as exc:
            raise PdfParseError(f"failed to extract text from PDF: {exc}") from exc
        finally:
            document.close()

        cleaned_pages = _clean_pages(raw_pages)

        warnings: list[ParseWarning] = []
        if two_column_pages > 0:
            warnings.append(ParseWarning.POSSIBLE_TWO_COLUMN_LAYOUT)
        warnings.extend(section_recovery.compute_quality_warnings(cleaned_pages))

        sections, section_warnings = section_recovery.recover_sections(
            cleaned_pages, paper_id=paper_id, paper_version_id=paper_version_id
        )
        warnings.extend(section_warnings)

        full_text = "\n\n".join(page.text for page in cleaned_pages).strip()
        if not full_text:
            raise PdfParseError(f"no extractable text found in PDF: {artifact_path}")

        return ParsedPaperDocument(
            paper_id=paper_id,
            paper_version_id=paper_version_id,
            pages=cleaned_pages,
            full_text=full_text,
            sections=sections,
            parser_name=self.name,
            parser_version=self.version,
            page_count=len(cleaned_pages),
            warnings=warnings,
        )


def _open_pdf(artifact_path: Path) -> "pymupdf.Document":
    try:
        return pymupdf.open(artifact_path)
    except pymupdf.FileDataError as exc:
        raise UnsupportedPdfError(f"file is not a readable PDF: {exc}") from exc
    except RuntimeError as exc:
        # PyMuPDF raises a plain RuntimeError for some corrupt/truncated files.
        raise UnsupportedPdfError(f"could not open PDF: {exc}") from exc


def _clean_pages(pages: list[ParsedPage]) -> list[ParsedPage]:
    normalized = [
        ParsedPage(page_number=page.page_number, text=normalization.clean_page_text(page.text))
        for page in pages
    ]
    headers, footers = normalization.detect_repeated_header_footer_lines(normalized)
    return normalization.remove_repeated_header_footer_lines(normalized, headers, footers)


def _looks_two_column(page: "pymupdf.Page") -> bool:
    """Conservative two-column heuristic: at least two substantial text
    blocks whose horizontal centers fall clearly left-of-center and at
    least two clearly right-of-center. Detection only -- reading order is
    never reconstructed (prompt #12)."""

    blocks = page.get_text("blocks")
    page_width = page.rect.width
    if page_width <= 0:
        return False

    left_blocks = 0
    right_blocks = 0
    for block in blocks:
        x0, _y0, x1, _y1, text = block[0], block[1], block[2], block[3], block[4]
        if not text.strip() or (x1 - x0) < _MIN_BLOCK_WIDTH:
            continue
        center_x = (x0 + x1) / 2
        if center_x < page_width * 0.45:
            left_blocks += 1
        elif center_x > page_width * 0.55:
            right_blocks += 1

    return left_blocks >= 2 and right_blocks >= 2
