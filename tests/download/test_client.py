"""Unit tests for `PdfDownloadClient`.

All requests go through `httpx.MockTransport` -- no real network access --
and the injected no-op `sleep` keeps retry-path tests fast.
"""

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import PdfDownloadError, PdfNotFoundError, PdfTimeoutError, PdfTooLargeError
from app.ingestion.download.client import PdfDownloadClient

_FAKE_PDF = b"%PDF-1.4\n%fake pdf content for testing\n%%EOF"


def _settings(**overrides) -> Settings:
    defaults = dict(PDF_DOWNLOAD_TIMEOUT_SECONDS=5, PDF_DOWNLOAD_MAX_RETRIES=2, MAX_PAPER_SIZE_MB=1)
    defaults.update(overrides)
    return Settings(**defaults)


def _client(handler, **settings_overrides) -> PdfDownloadClient:
    transport = httpx.MockTransport(handler)
    return PdfDownloadClient(
        _settings(**settings_overrides), transport=transport, sleep=lambda _: None
    )


class TestSuccessfulDownload:
    def test_valid_pdf_is_streamed_to_the_destination_path(self, tmp_path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=_FAKE_PDF, headers={"content-type": "application/pdf"}
            )

        dest = tmp_path / "paper.pdf.part"
        result = _client(handler).download("https://arxiv.org/pdf/2401.12345v1", dest)

        assert dest.read_bytes() == _FAKE_PDF
        assert result.bytes_downloaded == len(_FAKE_PDF)
        assert result.content_type == "application/pdf"

    def test_html_body_is_downloaded_as_is(self, tmp_path) -> None:
        # The client transports bytes; deciding whether they're actually a
        # PDF is `PdfAcquisitionService`'s job (signature check), not the
        # client's.
        html_body = b"<html><body>Not Found</body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=html_body, headers={"content-type": "text/html"})

        dest = tmp_path / "paper.pdf.part"
        result = _client(handler).download("https://arxiv.org/pdf/2401.12345v1", dest)

        assert dest.read_bytes() == html_body
        assert result.content_type == "text/html"

    def test_missing_content_type_header_still_downloads(self, tmp_path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_FAKE_PDF)

        dest = tmp_path / "paper.pdf.part"
        result = _client(handler).download("https://arxiv.org/pdf/2401.12345v1", dest)

        assert result.bytes_downloaded == len(_FAKE_PDF)
        assert result.content_type is None


class TestEmptyBody:
    def test_empty_body_downloads_as_a_zero_byte_file(self, tmp_path) -> None:
        # Rejecting a zero-byte file is the service's validation job, not
        # the client's -- the client just faithfully reports 0 bytes.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        dest = tmp_path / "paper.pdf.part"
        result = _client(handler).download("https://arxiv.org/pdf/2401.12345v1", dest)

        assert result.bytes_downloaded == 0
        assert dest.read_bytes() == b""


class TestOversizedDownload:
    def test_exceeding_the_size_limit_aborts_and_cleans_up_the_partial_file(self, tmp_path) -> None:
        big_body = b"x" * (2 * 1024 * 1024)  # 2MB, over the 1MB test limit

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=big_body)

        dest = tmp_path / "paper.pdf.part"
        with pytest.raises(PdfTooLargeError):
            _client(handler, MAX_PAPER_SIZE_MB=1).download(
                "https://arxiv.org/pdf/2401.12345v1", dest
            )

        assert not dest.exists()

    def test_oversized_download_is_not_retried(self, tmp_path) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            return httpx.Response(200, content=b"x" * (2 * 1024 * 1024))

        dest = tmp_path / "paper.pdf.part"
        with pytest.raises(PdfTooLargeError):
            _client(handler, MAX_PAPER_SIZE_MB=1, PDF_DOWNLOAD_MAX_RETRIES=2).download(
                "https://arxiv.org/pdf/x", dest
            )

        assert attempts["count"] == 1


class TestNotFound:
    def test_404_is_not_retried(self, tmp_path) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            return httpx.Response(404)

        dest = tmp_path / "paper.pdf.part"
        with pytest.raises(PdfNotFoundError):
            _client(handler, PDF_DOWNLOAD_MAX_RETRIES=2).download(
                "https://arxiv.org/pdf/missing", dest
            )

        assert attempts["count"] == 1


class TestTimeout:
    def test_exhausted_retries_raise_pdf_timeout_error(self, tmp_path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out", request=request)

        dest = tmp_path / "paper.pdf.part"
        with pytest.raises(PdfTimeoutError):
            _client(handler, PDF_DOWNLOAD_MAX_RETRIES=1).download("https://arxiv.org/pdf/x", dest)

    def test_succeeds_after_a_transient_timeout(self, tmp_path) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.TimeoutException("timed out", request=request)
            return httpx.Response(200, content=_FAKE_PDF)

        dest = tmp_path / "paper.pdf.part"
        result = _client(handler, PDF_DOWNLOAD_MAX_RETRIES=2).download(
            "https://arxiv.org/pdf/x", dest
        )

        assert result.bytes_downloaded == len(_FAKE_PDF)
        assert attempts["count"] == 2


class TestRateLimiting:
    def test_429_is_retried_then_succeeds(self, tmp_path) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(429)
            return httpx.Response(200, content=_FAKE_PDF)

        dest = tmp_path / "paper.pdf.part"
        result = _client(handler, PDF_DOWNLOAD_MAX_RETRIES=2).download(
            "https://arxiv.org/pdf/x", dest
        )

        assert result.bytes_downloaded == len(_FAKE_PDF)


class TestServerError:
    def test_500_is_retried_then_succeeds(self, tmp_path) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(500)
            return httpx.Response(200, content=_FAKE_PDF)

        dest = tmp_path / "paper.pdf.part"
        result = _client(handler, PDF_DOWNLOAD_MAX_RETRIES=2).download(
            "https://arxiv.org/pdf/x", dest
        )

        assert result.bytes_downloaded == len(_FAKE_PDF)

    def test_exhausted_retries_raise_pdf_download_error(self, tmp_path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        dest = tmp_path / "paper.pdf.part"
        with pytest.raises(PdfDownloadError):
            _client(handler, PDF_DOWNLOAD_MAX_RETRIES=1).download("https://arxiv.org/pdf/x", dest)


class TestPartialConnectionFailure:
    def test_transport_error_is_retried_then_succeeds(self, tmp_path) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.ConnectError("connection reset", request=request)
            return httpx.Response(200, content=_FAKE_PDF)

        dest = tmp_path / "paper.pdf.part"
        result = _client(handler, PDF_DOWNLOAD_MAX_RETRIES=2).download(
            "https://arxiv.org/pdf/x", dest
        )

        assert result.bytes_downloaded == len(_FAKE_PDF)

    def test_exhausted_transport_error_retries_raise(self, tmp_path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection reset", request=request)

        dest = tmp_path / "paper.pdf.part"
        with pytest.raises(PdfDownloadError):
            _client(handler, PDF_DOWNLOAD_MAX_RETRIES=1).download("https://arxiv.org/pdf/x", dest)
