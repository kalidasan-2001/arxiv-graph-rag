"""Paper discovery service: the only thing FastAPI routes should call.

Orchestrates: validate request -> call `ArxivClient` -> deduplicate ->
normalize -> convert to domain `Paper`/`PaperVersion` -> optionally persist
-> return results. Never triggers ingestion (CLAUDE.md's DISCOVER stage is
distinct from DOWNLOAD/PARSE/...) and never touches a PDF.
"""

import logging
import time
from datetime import datetime
from typing import Protocol

from app.core.config import Settings
from app.core.exceptions import InvalidSearchQueryError
from app.ingestion.discovery.models import ArxivPaperResult, PaperDiscoveryResult, PaperSearchQuery
from app.ingestion.discovery.normalization import (
    arxiv_result_to_paper,
    arxiv_result_to_paper_version,
)
from app.storage.postgres.repositories.papers import PaperRepository

logger = logging.getLogger(__name__)


class SupportsArxivSearch(Protocol):
    """Structural contract `PaperDiscoveryService` depends on.

    Lets tests substitute a fake client without inheriting from
    `ArxivClient` -- the service only ever needs this one method.
    """

    def search(self, query: PaperSearchQuery) -> list[ArxivPaperResult]: ...


class PaperDiscoveryService:
    """Search arXiv, normalize results, and (by default) persist metadata.

    Persistence policy (prompt #21, "Recommended"): search automatically
    upserts discovered paper metadata into PostgreSQL. It never creates an
    `ingestion_jobs` row and never changes ingestion state -- discovery and
    ingestion are deliberately separate lifecycle stages (CLAUDE.md #18 in
    this prompt: DISCOVERED != INGESTED). Pass `paper_repository=None` to
    search without persisting (e.g. the live smoke-test script).
    """

    def __init__(
        self,
        settings: Settings,
        arxiv_client: SupportsArxivSearch,
        paper_repository: PaperRepository | None = None,
    ) -> None:
        self._settings = settings
        self._client = arxiv_client
        self._repository = paper_repository

    def search(self, query: PaperSearchQuery) -> list[PaperDiscoveryResult]:
        resolved_max_results = self._resolve_max_results(query.max_results)
        resolved_query = query.model_copy(update={"max_results": resolved_max_results})

        started = time.monotonic()
        try:
            raw_results = self._client.search(resolved_query)
        except Exception:
            duration = time.monotonic() - started
            logger.warning(
                "arxiv discovery search failed query=%r requested_limit=%d "
                "duration_seconds=%.3f status=error",
                query.query,
                resolved_max_results,
                duration,
            )
            raise

        deduplicated = self._deduplicate(raw_results)
        filtered = self._filter_by_date_range(
            deduplicated, query.published_after, query.published_before
        )

        results: list[PaperDiscoveryResult] = []
        persisted_count = 0
        for arxiv_result in filtered:
            paper = arxiv_result_to_paper(arxiv_result)
            already_known = (
                self._repository is not None
                and self._repository.get_by_source(paper.source, paper.source_id) is not None
            )

            if self._repository is not None:
                paper = self._repository.upsert_paper(paper)
                persisted_count += 1
                version = arxiv_result_to_paper_version(arxiv_result, paper_id=paper.paper_id)
                if version is not None:
                    self._repository.get_or_create_paper_version(version)

            results.append(
                PaperDiscoveryResult(
                    paper=paper,
                    latest_version=arxiv_result.version,
                    already_known=already_known,
                )
            )

        duration = time.monotonic() - started
        logger.info(
            "arxiv discovery search completed query=%r requested_limit=%d "
            "returned_count=%d normalized_count=%d persisted_count=%d "
            "duration_seconds=%.3f status=ok",
            query.query,
            resolved_max_results,
            len(raw_results),
            len(results),
            persisted_count,
            duration,
        )
        return results

    def _resolve_max_results(self, requested: int | None) -> int:
        """Apply the configured default/limit.

        Policy: reject requests over the configured hard limit rather than
        silently clamping them (CLAUDE.md #43, no silent fallbacks) -- a
        caller asking for more than allowed should get an explicit error,
        not a quietly-truncated result set.
        """

        if requested is None:
            return self._settings.ARXIV_DEFAULT_MAX_RESULTS
        if requested > self._settings.ARXIV_MAX_RESULTS_LIMIT:
            raise InvalidSearchQueryError(
                f"max_results ({requested}) exceeds the configured limit "
                f"({self._settings.ARXIV_MAX_RESULTS_LIMIT})"
            )
        return requested

    @staticmethod
    def _deduplicate(results: list[ArxivPaperResult]) -> list[ArxivPaperResult]:
        """Deduplicate by `source_id`, keeping the highest version seen.

        arXiv's search API shouldn't normally return duplicates, but this
        guards against it deterministically rather than trusting the
        external response as-is (CLAUDE.md #36). Different, unrelated
        source ids are never merged -- only repeats of the same source_id.
        """

        best_by_id: dict[str, ArxivPaperResult] = {}
        order: list[str] = []
        for result in results:
            if result.source_id not in best_by_id:
                order.append(result.source_id)
                best_by_id[result.source_id] = result
            elif _version_rank(result.version) > _version_rank(best_by_id[result.source_id].version):
                best_by_id[result.source_id] = result
        return [best_by_id[source_id] for source_id in order]

    @staticmethod
    def _filter_by_date_range(
        results: list[ArxivPaperResult],
        published_after: datetime | None,
        published_before: datetime | None,
    ) -> list[ArxivPaperResult]:
        """Filter the already-retrieved page by publish date.

        arXiv's `search_query` syntax does support a `submittedDate:[..]`
        range filter, but this stage deliberately does not use it -- the
        date-range query format is easy to get subtly wrong, and this
        prompt explicitly sanctions client-side filtering as a documented
        fallback. This filters only the page of results already retrieved;
        it is NOT exhaustive over all matching papers on arXiv if more
        pages exist upstream (see docs/ARCHITECTURE.md "Paper Discovery").
        """

        if published_after is None and published_before is None:
            return results

        def in_range(result: ArxivPaperResult) -> bool:
            if result.published_at is None:
                return False
            if published_after is not None and result.published_at < published_after:
                return False
            if published_before is not None and result.published_at > published_before:
                return False
            return True

        return [r for r in results if in_range(r)]


def _version_rank(version: str | None) -> int:
    if not version:
        return -1
    digits = "".join(ch for ch in version if ch.isdigit())
    return int(digits) if digits else -1
