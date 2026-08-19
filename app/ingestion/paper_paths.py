"""Shared deterministic paper-version storage-path helpers.

Both raw PDF storage (`app.ingestion.download.storage.PaperStorage`) and
parsed-artifact storage (`app.ingestion.parsing.storage.ParsedArtifactStorage`)
store their file at the same paper-version directory --
``{root}/{source}/{source_id}/{version}/`` -- just with a different
filename, so that shared layout and its path-safety validation live in
exactly one place rather than two copies drifting apart.
"""

import re
from pathlib import Path

from app.core.exceptions import InvalidStoragePathError

# A single safe path segment: alphanumeric, `.`, `_`, `-` only, and never
# just `.` or `..` (excluded because they don't match "at least one
# alphanumeric" at each end).
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def sanitize_path_component(value: str, *, label: str) -> str:
    if not value or not _SAFE_COMPONENT.match(value):
        raise InvalidStoragePathError(f"unsafe storage path component for {label}: {value!r}")
    return value


def paper_version_directory(root: Path, *, source: str, source_id: str, version: str) -> Path:
    """The deterministic directory for one paper version's artifacts.

    Both `paper.pdf` and `parsed.json` live here, keyed by
    `(source, source_id, version)`, validated against path traversal.
    """

    source = sanitize_path_component(source, label="source")
    source_id = sanitize_path_component(source_id, label="source_id")
    version = sanitize_path_component(version, label="version")
    path = (root / source / source_id / version).resolve()

    if root not in path.parents:
        # Defense in depth: the sanitization above should already make
        # this unreachable, but a storage path must never be trusted to
        # stay inside its root without an explicit check.
        raise InvalidStoragePathError(f"resolved path escaped storage root: {path}")
    return path
