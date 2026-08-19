"""Shared SHA-256 file-checksum helper.

Used by both PDF acquisition (Prompt 4, verifying the downloaded/stored
raw PDF) and parsing (Prompt 5, re-verifying that PDF before parsing it).
"""

import hashlib
from pathlib import Path


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream-hash so large files are never loaded fully into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
