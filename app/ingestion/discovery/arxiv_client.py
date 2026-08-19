"""Low-level arXiv HTTP client.

Talks to the public arXiv API (``https://export.arxiv.org/api/query``, an
Atom feed) directly over HTTP via `httpx`, rather than adding a third-party
`arxiv` package: the Atom format is simple enough that stdlib
`xml.etree.ElementTree` parses it cleanly, `httpx` is already a project
dependency, and avoiding an extra dependency keeps this external boundary
small (CLAUDE.md #30, Dependency Discipline).

Returns only `ArxivPaperResult` DTOs -- callers never see raw XML, an
`httpx.Response`, or any other vendor-specific object (CLAUDE.md #15).
"""

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache

import httpx

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ArxivRateLimitError,
    ArxivResponseError,
    ArxivServiceError,
    ArxivTimeoutError,
)
from app.ingestion.discovery import normalization
from app.ingestion.discovery.models import ArxivPaperResult, PaperSearchQuery, SortBy, SortOrder

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_USER_AGENT = "arxiv-graph-rag/0.1"
_BACKOFF_SECONDS = 1.0
_RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"

_SORT_BY_PARAM = {
    SortBy.RELEVANCE: "relevance",
    SortBy.SUBMITTED_DATE: "submittedDate",
    SortBy.LAST_UPDATED_DATE: "lastUpdatedDate",
}
_SORT_ORDER_PARAM = {
    SortOrder.ASCENDING: "ascending",
    SortOrder.DESCENDING: "descending",
}


class ArxivClient:
    """Thin HTTP + Atom-parsing boundary around the public arXiv search API."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._max_retries = settings.ARXIV_MAX_RETRIES
        self._sleep = sleep
        self._client = httpx.Client(
            timeout=settings.ARXIV_REQUEST_TIMEOUT_SECONDS,
            transport=transport,
            headers={"User-Agent": _USER_AGENT},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ArxivClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def search(self, query: PaperSearchQuery) -> list[ArxivPaperResult]:
        """Execute a search request and return normalized results.

        `query.max_results` must already be resolved to a concrete positive
        int -- resolving the configured default/limit is
        `PaperDiscoveryService`'s job, not this client's.
        """

        if query.max_results is None:
            raise ValueError("query.max_results must be resolved before calling search()")

        params = self._build_params(query)
        response_text = self._get_with_retries(params)
        try:
            return _parse_feed(response_text)
        except ValueError as exc:
            raise ArxivResponseError(f"could not parse arXiv response: {exc}") from exc

    def _get_with_retries(self, params: dict[str, str]) -> str:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._client.get(_ARXIV_API_URL, params=params)
            except httpx.TimeoutException as exc:
                if attempt <= self._max_retries:
                    self._sleep(_BACKOFF_SECONDS * attempt)
                    continue
                raise ArxivTimeoutError(
                    f"arXiv request timed out after {attempt} attempt(s)"
                ) from exc
            except httpx.TransportError as exc:
                if attempt <= self._max_retries:
                    self._sleep(_BACKOFF_SECONDS * attempt)
                    continue
                raise ArxivServiceError(f"network error contacting arXiv: {exc}") from exc

            if response.status_code == 429:
                if attempt <= self._max_retries:
                    self._sleep(_BACKOFF_SECONDS * attempt)
                    continue
                raise ArxivRateLimitError("arXiv rate limit exceeded (HTTP 429)")

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt <= self._max_retries:
                self._sleep(_BACKOFF_SECONDS * attempt)
                continue

            if response.status_code >= 400:
                raise ArxivServiceError(f"arXiv returned HTTP {response.status_code}")

            return response.text

    @staticmethod
    def _build_params(query: PaperSearchQuery) -> dict[str, str]:
        # V1 keyword search: the whole normalized query is treated as one
        # phrase searched across all fields (`all:"..."`). Advanced arXiv
        # query-language passthrough (explicit ti:/abs:/AND/OR terms) is
        # not supported -- keeps this a predictable keyword-search boundary
        # rather than a query-language proxy.
        terms = [f'all:"{query.query}"']
        for category in query.categories:
            terms.append(f"cat:{category}")

        return {
            "search_query": " AND ".join(terms),
            "start": str(query.start),
            "max_results": str(query.max_results),
            "sortBy": _SORT_BY_PARAM[query.sort_by],
            "sortOrder": _SORT_ORDER_PARAM[query.sort_order],
        }


def _parse_feed(xml_text: str) -> list[ArxivPaperResult]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"arXiv response was not valid XML: {exc}") from exc

    return [_parse_entry(entry) for entry in root.findall(f"{_ATOM_NS}entry")]


def _parse_entry(entry: ET.Element) -> ArxivPaperResult:
    raw_id = _text(entry, f"{_ATOM_NS}id")
    if not raw_id:
        raise ValueError("arXiv entry is missing an <id>")
    source_id, version = normalization.normalize_arxiv_id(raw_id)

    authors = [_text(author, f"{_ATOM_NS}name") for author in entry.findall(f"{_ATOM_NS}author")]
    categories = [cat.get("term") for cat in entry.findall(f"{_ATOM_NS}category")]

    primary_category_el = entry.find(f"{_ARXIV_NS}primary_category")
    primary_category = (
        primary_category_el.get("term") if primary_category_el is not None else None
    )

    pdf_url: str | None = None
    entry_url: str | None = None
    for link in entry.findall(f"{_ATOM_NS}link"):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
        elif link.get("rel") == "alternate":
            entry_url = link.get("href")

    published_raw = _text(entry, f"{_ATOM_NS}published")
    updated_raw = _text(entry, f"{_ATOM_NS}updated")

    return ArxivPaperResult(
        source_id=source_id,
        version=version,
        title=_text(entry, f"{_ATOM_NS}title") or "",
        abstract=_text(entry, f"{_ATOM_NS}summary"),
        authors=[a for a in authors if a],
        categories=[c for c in categories if c],
        primary_category=primary_category,
        published_at=_parse_optional_datetime(published_raw),
        updated_at=_parse_optional_datetime(updated_raw),
        pdf_url=pdf_url,
        entry_url=entry_url,
    )


def _parse_optional_datetime(raw: str | None) -> datetime | None:
    return normalization.parse_arxiv_datetime(raw) if raw else None


def _text(element: ET.Element, tag: str) -> str | None:
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


@lru_cache
def get_arxiv_client() -> ArxivClient:
    """FastAPI dependency: a process-wide cached `ArxivClient`.

    Cached like `get_session_factory()` so the underlying `httpx` connection
    pool is reused across requests rather than rebuilt per call.
    """

    return ArxivClient(get_settings())
