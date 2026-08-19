"""Unit tests for SHA-256 artifact checksum calculation.

`sha256_file` (`app.ingestion.checksums`) is shared by both PDF
acquisition (Prompt 4) and parsing (Prompt 5) artifact verification.
"""

import hashlib

from app.ingestion.checksums import sha256_file


class TestChecksumCalculation:
    def test_same_bytes_produce_the_same_checksum(self, tmp_path) -> None:
        path_a = tmp_path / "a.pdf"
        path_b = tmp_path / "b.pdf"
        content = b"%PDF-1.4 identical content"
        path_a.write_bytes(content)
        path_b.write_bytes(content)

        assert sha256_file(path_a) == sha256_file(path_b)

    def test_different_bytes_produce_different_checksums(self, tmp_path) -> None:
        path_a = tmp_path / "a.pdf"
        path_b = tmp_path / "b.pdf"
        path_a.write_bytes(b"%PDF-1.4 content A")
        path_b.write_bytes(b"%PDF-1.4 content B")

        assert sha256_file(path_a) != sha256_file(path_b)

    def test_matches_the_hashlib_reference_implementation(self, tmp_path) -> None:
        path = tmp_path / "a.pdf"
        content = b"%PDF-1.4 " + b"some pdf-shaped bytes " * 50
        path.write_bytes(content)

        assert sha256_file(path) == hashlib.sha256(content).hexdigest()

    def test_streaming_read_handles_content_larger_than_one_internal_chunk(self, tmp_path) -> None:
        path = tmp_path / "big.pdf"
        # ~2MB, comfortably larger than sha256_file's 1MB internal chunk
        # size, so this exercises the multi-chunk read loop.
        content = b"%PDF-1.4 " + (b"0123456789" * 200_000)
        path.write_bytes(content)

        assert sha256_file(path) == hashlib.sha256(content).hexdigest()

    def test_empty_file_has_the_well_known_empty_sha256(self, tmp_path) -> None:
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"")

        assert sha256_file(path) == hashlib.sha256(b"").hexdigest()
