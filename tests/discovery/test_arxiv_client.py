"""Unit tests for `ArxivClient`.

All requests go through `httpx.MockTransport` -- no real network access,
and the injected no-op `sleep` keeps retry-path tests fast.
"""

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import (
    ArxivRateLimitError,
    ArxivResponseError,
    ArxivServiceError,
    ArxivTimeoutError,
)
from app.ingestion.discovery.arxiv_client import ArxivClient
from app.ingestion.discovery.models import PaperSearchQuery

_ONE_ENTRY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <title>Graph RAG: A Survey</title>
    <summary>An abstract about graph-based retrieval augmented generation.</summary>
    <published>2024-01-15T18:30:00Z</published>
    <updated>2024-01-20T09:00:00Z</updated>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.AI"/>
    <category term="cs.CL"/>
    <arxiv:primary_category term="cs.AI"/>
    <link rel="alternate" href="http://arxiv.org/abs/2401.12345v2"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2"/>
  </entry>
</feed>
"""

_ZERO_ENTRY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""

_MALFORMED_FEED = "<feed><entry><title>unterminated"


def _settings(**overrides) -> Settings:
    defaults = dict(ARXIV_REQUEST_TIMEOUT_SECONDS=5, ARXIV_MAX_RETRIES=2)
    defaults.update(overrides)
    return Settings(**defaults)


def _query(**overrides) -> PaperSearchQuery:
    defaults = dict(query="graph rag", max_results=10)
    defaults.update(overrides)
    return PaperSearchQuery(**defaults)


def _client(handler, **settings_overrides) -> ArxivClient:
    transport = httpx.MockTransport(handler)
    return ArxivClient(_settings(**settings_overrides), transport=transport, sleep=lambda _: None)


class TestSuccessfulResponse:
    def test_parses_entries_into_results(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=_ONE_ENTRY_FEED)

        client = _client(handler)
        results = client.search(_query())

        assert len(results) == 1
        result = results[0]
        assert result.source_id == "2401.12345"
        assert result.version == "v2"
        assert result.title == "Graph RAG: A Survey"
        assert result.authors == ["Alice Smith", "Bob Jones"]
        assert result.categories == ["cs.AI", "cs.CL"]
        assert result.pdf_url == "http://arxiv.org/pdf/2401.12345v2"

    def test_zero_results_returns_empty_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=_ZERO_ENTRY_FEED)

        client = _client(handler)
        assert client.search(_query()) == []

    def test_search_query_and_pagination_params_are_sent(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, text=_ZERO_ENTRY_FEED)

        client = _client(handler)
        client.search(_query(query="graph rag", start=10, max_results=5, categories=["cs.AI"]))

        assert captured["params"]["search_query"] == 'all:"graph rag" AND cat:cs.AI'
        assert captured["params"]["start"] == "10"
        assert captured["params"]["max_results"] == "5"

    def test_max_results_must_be_resolved_before_calling(self) -> None:
        client = _client(lambda request: httpx.Response(200, text=_ZERO_ENTRY_FEED))
        with pytest.raises(ValueError):
            client.search(_query(max_results=None))


class TestTimeout:
    def test_exhausted_retries_raise_arxiv_timeout_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out", request=request)

        client = _client(handler, ARXIV_MAX_RETRIES=1)
        with pytest.raises(ArxivTimeoutError):
            client.search(_query())

    def test_succeeds_after_a_transient_timeout(self) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.TimeoutException("timed out", request=request)
            return httpx.Response(200, text=_ONE_ENTRY_FEED)

        client = _client(handler, ARXIV_MAX_RETRIES=2)
        results = client.search(_query())

        assert len(results) == 1
        assert attempts["count"] == 2


class TestRateLimiting:
    def test_exhausted_retries_raise_arxiv_rate_limit_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="rate limited")

        client = _client(handler, ARXIV_MAX_RETRIES=1)
        with pytest.raises(ArxivRateLimitError):
            client.search(_query())

    def test_succeeds_after_a_transient_429(self) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(429, text="rate limited")
            return httpx.Response(200, text=_ONE_ENTRY_FEED)

        client = _client(handler, ARXIV_MAX_RETRIES=2)
        results = client.search(_query())

        assert len(results) == 1


class TestTemporaryServerError:
    def test_succeeds_after_a_transient_503(self) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(503, text="unavailable")
            return httpx.Response(200, text=_ONE_ENTRY_FEED)

        client = _client(handler, ARXIV_MAX_RETRIES=2)
        results = client.search(_query())

        assert len(results) == 1

    def test_exhausted_retries_raise_arxiv_service_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="internal error")

        client = _client(handler, ARXIV_MAX_RETRIES=1)
        with pytest.raises(ArxivServiceError):
            client.search(_query())

    def test_client_error_is_not_retried(self) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            return httpx.Response(400, text="bad request")

        client = _client(handler, ARXIV_MAX_RETRIES=2)
        with pytest.raises(ArxivServiceError):
            client.search(_query())

        # 400 is not in the retryable set -- must fail on the first attempt.
        assert attempts["count"] == 1


class TestMalformedResponse:
    def test_invalid_xml_raises_arxiv_response_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=_MALFORMED_FEED)

        client = _client(handler)
        with pytest.raises(ArxivResponseError):
            client.search(_query())
