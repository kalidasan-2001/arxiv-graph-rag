"""External-boundary and request/result DTOs for arXiv discovery.

`ArxivPaperResult` represents arXiv data *before* conversion into the core
`app.domain.papers.Paper` model -- nothing outside `app/ingestion/discovery/`
should depend on it. `PaperSearchQuery` is this stage's request model;
`PaperDiscoveryResult` is what `PaperDiscoveryService.search()` returns.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.ids import normalize_whitespace
from app.domain.papers import Paper


class SortBy(str, Enum):
    """Sort field, mapped explicitly onto arXiv's `sortBy` query parameter."""

    RELEVANCE = "relevance"
    SUBMITTED_DATE = "submitted_date"
    LAST_UPDATED_DATE = "last_updated_date"


class SortOrder(str, Enum):
    """Sort direction, mapped explicitly onto arXiv's `sortOrder` query parameter."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class ArxivPaperResult(BaseModel):
    """A single normalized arXiv search result, before conversion to `Paper`.

    "Normalized" here means: parsed out of Atom XML into plain fields.
    Author/category deduplication and text cleanup happen in
    `normalization.py`, not here -- this DTO is a faithful, minimally
    processed transcription of one `<entry>`.
    """

    source_id: str
    version: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    pdf_url: str | None = None
    entry_url: str | None = None


class PaperSearchQuery(BaseModel):
    """A validated request to search arXiv.

    `max_results=None` means "use the configured default" -- resolved by
    `PaperDiscoveryService` (which knows about `Settings`), not by this
    model, so this DTO stays independent of application configuration and
    easy to unit test in isolation.
    """

    query: str
    max_results: int | None = None
    start: int = 0
    sort_by: SortBy = SortBy.RELEVANCE
    sort_order: SortOrder = SortOrder.DESCENDING
    categories: list[str] = Field(default_factory=list)
    published_after: datetime | None = None
    published_before: datetime | None = None

    @field_validator("query")
    @classmethod
    def _query_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("search query must not be blank")
        return normalized

    @field_validator("start")
    @classmethod
    def _start_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("start must be >= 0")
        return value

    @field_validator("max_results")
    @classmethod
    def _max_results_positive_if_set(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_results must be > 0")
        return value

    @field_validator("categories")
    @classmethod
    def _normalize_categories(cls, values: list[str]) -> list[str]:
        return [normalized for v in values if (normalized := normalize_whitespace(v))]

    @model_validator(mode="after")
    def _date_range_ordered(self) -> "PaperSearchQuery":
        if (
            self.published_after is not None
            and self.published_before is not None
            and self.published_after > self.published_before
        ):
            raise ValueError("published_after must be <= published_before")
        return self


class PaperDiscoveryResult(BaseModel):
    """One discovered paper plus the discovery-specific context an API caller needs."""

    paper: Paper
    latest_version: str | None = None
    already_known: bool = False
