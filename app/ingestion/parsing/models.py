"""DTOs for the parsing stage: raw extracted pages and the resulting
structured document.

Not domain models -- `ParsedPage`/`ParsedPaperDocument` are parsing-stage
DTOs, mirroring how `ArxivPaperResult` (discovery) and `AcquisitionResult`
(download) stay local to their own stage rather than living in
`app.domain`. `ParsedPaperDocument.sections` reuses the existing domain
`PaperSection` model directly -- no new identity system.
"""

from enum import Enum

from pydantic import BaseModel, Field

from app.domain.papers import PaperSection


class ParseWarning(str, Enum):
    """Deterministic, non-fatal parsing quality signals (prompt #34).

    Parsing is not binary success/failure -- these describe *why* a parse
    might be lower-confidence without necessarily failing it.
    """

    NO_SECTION_HEADINGS_DETECTED = "no_section_headings_detected"
    POSSIBLE_TWO_COLUMN_LAYOUT = "possible_two_column_layout"
    NO_ABSTRACT_DETECTED = "no_abstract_detected"
    NO_REFERENCES_DETECTED = "no_references_detected"
    LOW_TEXT_DENSITY = "low_text_density"
    EMPTY_PAGE_DETECTED = "empty_page_detected"


class ParsedPage(BaseModel):
    """One page's cleaned, extracted plain text."""

    page_number: int  # 1-indexed, matching how humans reference PDF pages
    text: str


class ParsedPaperDocument(BaseModel):
    """The full structured output of parsing one paper version.

    `source_pdf_checksum` is the raw PDF's SHA-256 *at the time this parse
    was produced* -- distinct from `PaperVersion.checksum` (which always
    reflects the PDF's *current* state) so reconciliation can detect "the
    underlying PDF changed since this parse was made" (prompt #29).
    """

    paper_id: str
    paper_version_id: str
    pages: list[ParsedPage]
    full_text: str
    sections: list[PaperSection] = Field(default_factory=list)
    parser_name: str
    parser_version: str
    source_pdf_checksum: str = ""
    page_count: int
    warnings: list[ParseWarning] = Field(default_factory=list)
