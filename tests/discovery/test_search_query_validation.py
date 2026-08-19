"""Unit tests for `PaperSearchQuery` validation and
`PaperDiscoveryService._resolve_max_results`'s config-dependent bounds.
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import InvalidSearchQueryError
from app.ingestion.discovery.models import PaperSearchQuery, SortBy, SortOrder
from app.ingestion.discovery.service import PaperDiscoveryService


class TestPaperSearchQueryValidation:
    def test_empty_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperSearchQuery(query="   ")

    def test_max_results_zero_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperSearchQuery(query="graph rag", max_results=0)

    def test_max_results_negative_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperSearchQuery(query="graph rag", max_results=-5)

    def test_negative_start_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperSearchQuery(query="graph rag", start=-1)

    def test_inverted_date_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperSearchQuery(
                query="graph rag",
                published_after="2024-06-01T00:00:00Z",
                published_before="2024-01-01T00:00:00Z",
            )

    def test_unsupported_sort_by_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperSearchQuery(query="graph rag", sort_by="popularity")

    def test_unsupported_sort_order_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperSearchQuery(query="graph rag", sort_order="sideways")

    def test_valid_query_is_accepted(self) -> None:
        query = PaperSearchQuery(
            query="  graph rag  ",
            max_results=10,
            sort_by=SortBy.SUBMITTED_DATE,
            sort_order=SortOrder.ASCENDING,
        )
        assert query.query == "graph rag"
        assert query.max_results == 10


class TestResolveMaxResults:
    def _service(self, **settings_overrides) -> PaperDiscoveryService:
        defaults = dict(ARXIV_DEFAULT_MAX_RESULTS=20, ARXIV_MAX_RESULTS_LIMIT=100)
        defaults.update(settings_overrides)
        return PaperDiscoveryService(Settings(**defaults), arxiv_client=object())

    def test_none_resolves_to_configured_default(self) -> None:
        service = self._service()
        assert service._resolve_max_results(None) == 20

    def test_within_limit_is_returned_as_is(self) -> None:
        service = self._service()
        assert service._resolve_max_results(50) == 50

    def test_over_limit_is_rejected_not_clamped(self) -> None:
        service = self._service()
        with pytest.raises(InvalidSearchQueryError):
            service._resolve_max_results(500)

    def test_exactly_at_limit_is_accepted(self) -> None:
        service = self._service()
        assert service._resolve_max_results(100) == 100
