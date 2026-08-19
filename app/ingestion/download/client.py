"""Streaming PDF download client.

Streams to a caller-provided destination path via `httpx` (`stream()`, not
`.content`) so large PDFs are never held fully in memory. Enforces
`MAX_PAPER_SIZE_MB` *while* streaming -- aborting and deleting the partial
file the moment the limit is exceeded, not after the whole body has been
received -- and `PDF_DOWNLOAD_TIMEOUT_SECONDS` via httpx's own timeout.
Bounded retry mirrors `ArxivClient`'s pattern (same backoff shape, same
injectable `sleep` for fast tests) but only for genuinely transient
failures: HTTP 404, oversized responses, and other 4xx are never retried.

Redirects are not followed automatically -- the URL that reaches this
client has already been validated against a trusted-host allowlist
upstream (`PdfAcquisitionService`); silently following a redirect could
send the request somewhere that allowlist never approved.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.core.config import Settings, get_settings
from app.core.exceptions import PdfDownloadError, PdfNotFoundError, PdfTimeoutError, PdfTooLargeError

_USER_AGENT = "arxiv-graph-rag/0.1"
_BACKOFF_SECONDS = 1.0
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True)
class DownloadResult:
    bytes_downloaded: int
    content_type: str | None


class _RetryableDownloadError(Exception):
    """Internal signal: this attempt failed transiently and may be retried."""

    def __init__(self, wrapped: PdfDownloadError) -> None:
        self.wrapped = wrapped


class PdfDownloadClient:
    """Thin HTTP boundary that streams a PDF to disk."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._max_bytes = settings.MAX_PAPER_SIZE_MB * 1024 * 1024
        self._max_retries = settings.PDF_DOWNLOAD_MAX_RETRIES
        self._sleep = sleep
        self._client = httpx.Client(
            timeout=settings.PDF_DOWNLOAD_TIMEOUT_SECONDS,
            transport=transport,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PdfDownloadClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def download(self, url: str, dest_path: Path) -> DownloadResult:
        """Stream `url` to `dest_path`, retrying transient failures.

        `dest_path` is caller-chosen (always a `.part` temp path from
        `PaperStorage.get_temp_path` in practice) -- this client knows
        nothing about the paper-storage layout.
        """

        attempt = 0
        while True:
            attempt += 1
            try:
                return self._attempt(url, dest_path)
            except _RetryableDownloadError as exc:
                dest_path.unlink(missing_ok=True)
                if attempt <= self._max_retries:
                    self._sleep(_BACKOFF_SECONDS * attempt)
                    continue
                raise exc.wrapped  # noqa: B904 -- implicit chaining via __context__ is enough here

    def _attempt(self, url: str, dest_path: Path) -> DownloadResult:
        try:
            with self._client.stream("GET", url) as response:
                self._check_status(response)
                return self._stream_to_file(response, dest_path)
        except httpx.TimeoutException as exc:
            raise _RetryableDownloadError(
                PdfTimeoutError(f"PDF download timed out: {exc}")
            ) from exc
        except httpx.TransportError as exc:
            raise _RetryableDownloadError(
                PdfDownloadError(f"network error downloading PDF: {exc}")
            ) from exc

    @staticmethod
    def _check_status(response: httpx.Response) -> None:
        if response.status_code == 404:
            raise PdfNotFoundError(f"PDF not found (HTTP 404): {response.url}")
        if response.status_code == 429 or response.status_code in _RETRYABLE_STATUS_CODES:
            raise _RetryableDownloadError(
                PdfDownloadError(f"transient HTTP {response.status_code} downloading PDF")
            )
        if response.status_code >= 300:
            # Includes redirects (3xx): not followed automatically (see
            # module docstring), so a redirect response is itself an error
            # here, not a success to chase.
            raise PdfDownloadError(f"PDF download failed with HTTP {response.status_code}")

    def _stream_to_file(self, response: httpx.Response, dest_path: Path) -> DownloadResult:
        content_type = response.headers.get("content-type")
        bytes_downloaded = 0
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(dest_path, "wb") as fh:
                for chunk in response.iter_bytes(_CHUNK_SIZE):
                    bytes_downloaded += len(chunk)
                    if bytes_downloaded > self._max_bytes:
                        raise PdfTooLargeError(
                            f"PDF exceeds the configured {self._max_bytes}-byte limit"
                        )
                    fh.write(chunk)
        except PdfTooLargeError:
            dest_path.unlink(missing_ok=True)
            raise
        return DownloadResult(bytes_downloaded=bytes_downloaded, content_type=content_type)


def get_pdf_download_client() -> PdfDownloadClient:
    """FastAPI dependency: a fresh `PdfDownloadClient` per request.

    Unlike `get_arxiv_client`, not process-wide cached: PDF downloads are
    infrequent explicit actions (not a hot path), and each carries a
    request-specific timeout/retry configuration snapshot -- simplicity
    over the minor connection-reuse benefit here.
    """

    return PdfDownloadClient(get_settings())
