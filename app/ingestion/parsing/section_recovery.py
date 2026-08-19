"""Deterministic section-boundary detection and recovery (prompt #20).

No LLM, no layout/CV engine -- a heading is recognized either by a strong
*structural* signal (numbered/roman-numeral prefix on its own line) or, for
unnumbered headings, by an *exact* match against a known vocabulary of
scientific-paper section names. Anything else is left as ordinary body
text. This keeps the algorithm testable and its false-positive rate low,
at the cost of missing unconventional heading styles (documented
limitation, not a bug).

V1 scope: only top-level numbered headings ("1 Introduction", not
"1.1 Background") become section boundaries -- subsections stay part of
their parent section's text (prompt #16).
"""

import re

from app.domain.enums import SectionType
from app.domain.papers import PaperSection
from app.ingestion.parsing.models import ParsedPage, ParseWarning

# Top-level only: the numeric prefix must not contain a "." (that would be
# a subsection, e.g. "1.1 Background", which stays part of its parent).
_NUMBERED_HEADING = re.compile(
    r"^(?P<number>\d{1,2})\.?\s+(?P<title>[A-Z][A-Za-z][A-Za-z\s\-:]{2,60})$"
)
# Roman numerals require a trailing period ("I. Introduction") -- without
# it, a bare "I" or "V" is far too easy to confuse with real text.
_ROMAN_HEADING = re.compile(
    r"^(?P<number>[IVXLCDM]{1,6})\.\s+(?P<title>[A-Z][A-Za-z][A-Za-z\s\-:]{2,60})$"
)

# Tier 1: recognized scientific sections, mapped to a specific SectionType.
_SECTION_TYPE_KEYWORDS: dict[str, SectionType] = {
    "abstract": SectionType.ABSTRACT,
    "introduction": SectionType.INTRODUCTION,
    "related work": SectionType.RELATED_WORK,
    "related works": SectionType.RELATED_WORK,
    "method": SectionType.METHODOLOGY,
    "methods": SectionType.METHODOLOGY,
    "methodology": SectionType.METHODOLOGY,
    "approach": SectionType.METHODOLOGY,
    "experiments": SectionType.EXPERIMENTS,
    "experiment": SectionType.EXPERIMENTS,
    "experimental setup": SectionType.EXPERIMENTS,
    "experiment setup": SectionType.EXPERIMENTS,
    "evaluation": SectionType.EXPERIMENTS,
    "results": SectionType.RESULTS,
    "discussion": SectionType.DISCUSSION,
    "limitations": SectionType.LIMITATIONS,
    "limitation": SectionType.LIMITATIONS,
    "conclusion": SectionType.CONCLUSION,
    "conclusions": SectionType.CONCLUSION,
    "references": SectionType.REFERENCES,
    "bibliography": SectionType.REFERENCES,
}

# Tier 2: recognized as a real heading, but deliberately left as OTHER
# rather than forced into the initial ontology (prompt #19).
_OTHER_HEADING_KEYWORDS: frozenset[str] = frozenset(
    {
        "background",
        "implementation",
        "implementation details",
        "ablation study",
        "ablation studies",
        "ethics statement",
        "ethical considerations",
        "broader impact",
        "appendix",
        "acknowledgments",
        "acknowledgements",
    }
)


def _classify_heading_text(title: str) -> SectionType | None:
    """Map heading text to a `SectionType`, `OTHER`, or `None` (not a
    recognized heading at all)."""

    normalized = " ".join(title.strip().lower().split())
    if normalized in _SECTION_TYPE_KEYWORDS:
        return _SECTION_TYPE_KEYWORDS[normalized]
    if normalized in _OTHER_HEADING_KEYWORDS:
        return SectionType.OTHER
    return None


def detect_heading(line: str) -> tuple[str, SectionType] | None:
    """Return `(original_title, section_type)` if `line` looks like a
    top-level section heading, else `None`.

    Numbered/roman-numbered lines are trusted structurally even when their
    title text isn't in the vocabulary (preserved as `OTHER` -- prompt
    #19). Bare, unnumbered lines are only trusted when they exactly match
    the vocabulary, since without numbering the structural signal alone is
    too weak (prompt #11's "if confidence is low, preserve the text").
    """

    stripped = line.strip()
    if not stripped:
        return None

    match = _NUMBERED_HEADING.match(stripped)
    if match:
        title = match.group("title").strip()
        return title, (_classify_heading_text(title) or SectionType.OTHER)

    match = _ROMAN_HEADING.match(stripped)
    if match:
        title = match.group("title").strip()
        return title, (_classify_heading_text(title) or SectionType.OTHER)

    if len(stripped) <= 50:
        section_type = _classify_heading_text(stripped)
        if section_type is not None:
            return stripped, section_type

    return None


def recover_sections(
    pages: list[ParsedPage], *, paper_id: str, paper_version_id: str
) -> tuple[list[PaperSection], list[ParseWarning]]:
    """Detect headings across all pages and assemble section boundaries.

    Text appearing *before* the first detected heading (title, authors,
    affiliations -- already captured separately via arXiv discovery
    metadata) is not turned into a section of its own; V1 scope starts
    structuring content from the first recognized heading onward.
    """

    warnings: list[ParseWarning] = []

    flat_lines: list[tuple[int, str]] = [
        (page.page_number, line) for page in pages for line in page.text.split("\n")
    ]

    boundaries = [
        (idx, page_number, *detected)
        for idx, (page_number, line) in enumerate(flat_lines)
        if (detected := detect_heading(line)) is not None
    ]

    if not boundaries:
        warnings.append(ParseWarning.NO_SECTION_HEADINGS_DETECTED)
        full_text = "\n".join(text for _, text in flat_lines).strip()
        if not full_text or not pages:
            return [], warnings
        section = PaperSection.create(
            paper_id=paper_id,
            paper_version_id=paper_version_id,
            section_type=SectionType.OTHER,
            order=0,
            text=full_text,
            page_start=pages[0].page_number,
            page_end=pages[-1].page_number,
        )
        return [section], warnings

    sections: list[PaperSection] = []
    for i, (line_idx, page_number, title, section_type) in enumerate(boundaries):
        start_idx = line_idx + 1  # body text starts after the heading line
        end_idx = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(flat_lines)
        section_lines = flat_lines[start_idx:end_idx]
        text = "\n".join(text for _, text in section_lines).strip()
        if not text:
            continue  # heading with no body text -- skip rather than fabricate an empty section

        sections.append(
            PaperSection.create(
                paper_id=paper_id,
                paper_version_id=paper_version_id,
                section_type=section_type,
                title=title,
                order=len(sections),
                page_start=page_number,
                page_end=section_lines[-1][0],
                text=text,
            )
        )

    if not any(s.section_type == SectionType.ABSTRACT for s in sections):
        warnings.append(ParseWarning.NO_ABSTRACT_DETECTED)
    if not any(s.section_type == SectionType.REFERENCES for s in sections):
        warnings.append(ParseWarning.NO_REFERENCES_DETECTED)

    return sections, warnings


def compute_quality_warnings(pages: list[ParsedPage]) -> list[ParseWarning]:
    """Deterministic page-level quality signals, independent of section
    recovery (prompt #34/#35)."""

    warnings: list[ParseWarning] = []
    if any(not page.text.strip() for page in pages):
        warnings.append(ParseWarning.EMPTY_PAGE_DETECTED)

    total_chars = sum(len(page.text) for page in pages)
    avg_chars_per_page = total_chars / len(pages) if pages else 0
    # Threshold is deliberately low -- flags near-empty pages (e.g.
    # scanned/image-only PDFs with no embedded text layer), not merely
    # short ones. Not a substitute for OCR (prompt #38: none implemented).
    if avg_chars_per_page < 200:
        warnings.append(ParseWarning.LOW_TEXT_DENSITY)

    return warnings
