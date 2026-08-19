"""Pure normalization helpers for the arXiv discovery boundary.

No network calls, no persistence -- only deterministic text/id/date
transforms and DTO -> domain conversion. This is mechanical cleanup, not
semantic understanding: no summarization, no rewriting, no entity
inference, no LLM (CLAUDE.md #10's "no LLM for deterministic cleanup"
principle applies here too, even though #10 was written about chunking).
"""

import re
from datetime import datetime

from app.domain.ids import normalize_whitespace
from app.domain.papers import Paper, PaperVersion
from app.ingestion.discovery.models import ArxivPaperResult

_ARXIV_URL_PREFIX = re.compile(r"^https?://arxiv\.org/(abs|pdf)/", re.IGNORECASE)
_PDF_SUFFIX = re.compile(r"\.pdf$", re.IGNORECASE)
_VERSION_SUFFIX = re.compile(r"v(\d+)$", re.IGNORECASE)


def normalize_arxiv_id(raw: str) -> tuple[str, str | None]:
    """Normalize a raw arXiv identifier or URL into `(source_id, version)`.

    Accepts a bare id (``"2401.12345"``), a versioned id
    (``"2401.12345v2"``), or a full arXiv URL -- ``.../abs/...`` or the
    Atom feed's PDF-link form ``.../pdf/....pdf`` -- with or without a
    version suffix. `version` is returned as ``"v2"`` (already in the form
    `app.domain.ids.build_paper_version_id` expects), or `None` if the
    input carried no version suffix at all.

    Examples::

        normalize_arxiv_id("2401.12345")                              -> ("2401.12345", None)
        normalize_arxiv_id("2401.12345v2")                             -> ("2401.12345", "v2")
        normalize_arxiv_id("http://arxiv.org/abs/2401.12345v2")        -> ("2401.12345", "v2")
        normalize_arxiv_id("https://arxiv.org/abs/2401.12345")         -> ("2401.12345", None)
    """

    if not raw or not raw.strip():
        raise ValueError("arXiv id must not be blank")

    candidate = raw.strip()
    candidate = _ARXIV_URL_PREFIX.sub("", candidate)
    candidate = _PDF_SUFFIX.sub("", candidate)
    candidate = candidate.rstrip("/")

    version: str | None = None
    version_match = _VERSION_SUFFIX.search(candidate)
    if version_match:
        version = f"v{version_match.group(1)}"
        candidate = candidate[: version_match.start()]

    if not candidate:
        raise ValueError(f"could not extract an arXiv id from '{raw}'")

    return candidate, version


def normalize_search_query(raw: str) -> str:
    """Trim/collapse whitespace in a user-supplied search query.

    Raises `ValueError` if nothing meaningful remains -- callers map this
    to `InvalidSearchQueryError` at the service boundary.
    """

    normalized = normalize_whitespace(raw)
    if not normalized:
        raise ValueError("search query must not be blank")
    return normalized


def normalize_title(raw: str) -> str:
    """Collapse the internal line-wrapping whitespace arXiv titles often carry."""

    return normalize_whitespace(raw)


def normalize_abstract(raw: str) -> str:
    """Collapse whitespace in an arXiv abstract without altering its wording."""

    return normalize_whitespace(raw)


def normalize_authors(authors: list[str]) -> list[str]:
    """Trim whitespace and drop exact-duplicate author names, preserving order."""

    seen: set[str] = set()
    result: list[str] = []
    for author in authors:
        name = normalize_whitespace(author)
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def normalize_categories(categories: list[str]) -> list[str]:
    """Trim, deduplicate, and sort category codes for deterministic ordering."""

    seen: set[str] = set()
    result: list[str] = []
    for category in categories:
        code = normalize_whitespace(category)
        if not code or code in seen:
            continue
        seen.add(code)
        result.append(code)
    return sorted(result)


def parse_arxiv_datetime(raw: str) -> datetime:
    """Parse an arXiv Atom `<published>`/`<updated>` RFC 3339 timestamp."""

    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def arxiv_result_to_paper(result: ArxivPaperResult) -> Paper:
    """Convert a normalized arXiv result into the core domain `Paper`.

    `Paper.create` derives the stable `paper_id` from `("arxiv", source_id)`
    via `app.domain.ids.build_paper_id` -- the same builder every other
    stage uses, so no parallel arXiv-id system is introduced here.
    """

    return Paper.create(
        source="arxiv",
        source_id=result.source_id,
        title=normalize_title(result.title),
        abstract=normalize_abstract(result.abstract) if result.abstract else None,
        authors=normalize_authors(result.authors),
        categories=normalize_categories(result.categories),
        published_at=result.published_at,
        updated_at=result.updated_at,
        pdf_url=result.pdf_url,
    )


def arxiv_result_to_paper_version(
    result: ArxivPaperResult, *, paper_id: str
) -> PaperVersion | None:
    """Convert a normalized arXiv result into a `PaperVersion`, if it carries version info.

    Returns `None` when no version suffix could be determined -- discovery
    must not fabricate a version number arXiv didn't actually provide
    (prompt #17: "If arXiv provides version information reliably...").
    """

    if not result.version:
        return None
    return PaperVersion.create(
        paper_id=paper_id,
        version=result.version,
        source_version=result.version,
    )
