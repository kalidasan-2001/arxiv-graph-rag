"""Deterministic, conservative citation resolution (prompt #16/#17) --
no LLM. Only an explicit arXiv identifier is trusted enough to become a
`CITES` edge; everything else becomes an `UnresolvedCitation` rather than
risking a wrong edge from a fuzzy title/author guess.

Exact-normalized-title matching against the known paper catalog (the
prompt's other suggested trusted case) is intentionally **not**
implemented in V1 -- reliably isolating "the title" from an arbitrary
bibliography entry's free text is itself unreliable across citation
styles, and getting it wrong risks exactly the "wrong citation edge" this
stage is designed to avoid. Deferred; see docs/ARCHITECTURE.md.
"""

import re

from app.domain.ids import build_paper_id, normalize_whitespace
from app.ingestion.graph_extraction.models import UnresolvedCitation

# A reference entry typically opens with a bracketed/numbered marker:
# "[12] ...", "(3) ...", "12. ...". This is how the large majority of
# scientific bibliographies are formatted, and unambiguous to detect.
_ENTRY_MARKER_PATTERN = re.compile(r"^\s*(?:\[\d{1,4}\]|\(\d{1,4}\)|\d{1,4}\.)\s+", re.MULTILINE)

# Modern arXiv ids ("2401.12345", optionally "arXiv:" prefixed, optionally
# versioned) or legacy archive-style ids ("cs/0501001", "math.GT/0309136").
_ARXIV_ID_PATTERN = re.compile(
    r"(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
_TRAILING_VERSION_PATTERN = re.compile(r"v\d+$", re.IGNORECASE)

# Bounded so a malformed/giant "entry" (e.g. failed segmentation lumping
# the whole references section into one block) never gets stored whole
# (prompt #18: "do not store huge chunks redundantly").
_MAX_RAW_TEXT_LENGTH = 300


def split_reference_entries(references_text: str) -> list[str]:
    """Segment a REFERENCES section's raw text into individual bibliography
    entries. Numbered/bracketed markers are the primary signal; falls back
    to blank-line-separated blocks (some venues use unnumbered author-year
    style) when fewer than two markers are found -- still deterministic,
    just a weaker segmentation signal for that fallback case."""

    markers = list(_ENTRY_MARKER_PATTERN.finditer(references_text))
    if len(markers) >= 2:  # a single stray match isn't trustworthy evidence of real numbering
        entries = []
        for i, marker in enumerate(markers):
            start = marker.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(references_text)
            entry = normalize_whitespace(references_text[start:end])
            if entry:
                entries.append(entry)
        return entries

    blocks = re.split(r"\n\s*\n", references_text)
    return [normalized for block in blocks if (normalized := normalize_whitespace(block))]


def extract_arxiv_id(entry_text: str) -> str | None:
    """Return a normalized arXiv id (version suffix stripped -- `CITES`
    targets the logical paper, not one specific revision) or `None`."""

    match = _ARXIV_ID_PATTERN.search(entry_text)
    if match is None:
        return None
    return _TRAILING_VERSION_PATTERN.sub("", match.group(1))


def resolve_citations(
    references_text: str, *, source_chunk_id: str
) -> tuple[list[str], list[UnresolvedCitation]]:
    """Resolve a REFERENCES chunk's entries to arXiv paper ids where
    possible. Returns `(resolved_paper_ids, unresolved_citations)` --
    order-preserving, may contain duplicate paper ids if cited more than
    once (deduplicated by the caller when building relationships)."""

    resolved: list[str] = []
    unresolved: list[UnresolvedCitation] = []
    for entry in split_reference_entries(references_text):
        arxiv_id = extract_arxiv_id(entry)
        if arxiv_id is not None:
            resolved.append(build_paper_id("arxiv", arxiv_id))
        else:
            unresolved.append(
                UnresolvedCitation(
                    raw_text=entry[:_MAX_RAW_TEXT_LENGTH],
                    source_chunk_id=source_chunk_id,
                    reason="no explicit arXiv ID found in reference entry",
                )
            )
    return resolved, unresolved
