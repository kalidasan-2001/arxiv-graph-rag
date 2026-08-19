"""Local filesystem storage for raw PDF artifacts.

Deterministic layout::

    {PAPER_STORAGE_PATH}/{source}/{source_id}/{version}/paper.pdf

e.g. ``data/papers/arxiv/2401.12345/v2/paper.pdf`` -- the logical paper and
its version are both encoded in the path, so different versions never
collide or overwrite each other.

Path components (`source`, `source_id`, `version`) come only from
already-validated domain identifiers, never from a raw URL or paper
title, and are re-sanitized (via `app.ingestion.paper_paths`, shared with
`app.ingestion.parsing.storage.ParsedArtifactStorage`) before touching the
filesystem as defense in depth against an unexpected id shape from a
future source provider.
"""

from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.ingestion.paper_paths import paper_version_directory


class PaperStorage:
    """Filesystem-backed storage for raw PDF artifacts.

    Only the methods this stage actually needs -- not a generic blob-
    storage framework (CLAUDE.md #52).
    """

    _FINAL_NAME = "paper.pdf"
    _TEMP_SUFFIX = ".part"

    def __init__(self, settings: Settings) -> None:
        if not settings.PAPER_STORAGE_PATH:
            raise ConfigurationError("PAPER_STORAGE_PATH is not configured")
        self._root = Path(settings.PAPER_STORAGE_PATH).resolve()

    def get_path(self, *, source: str, source_id: str, version: str) -> Path:
        """The deterministic final artifact path for a paper version."""

        directory = paper_version_directory(
            self._root, source=source, source_id=source_id, version=version
        )
        return directory / self._FINAL_NAME

    def get_temp_path(self, *, source: str, source_id: str, version: str) -> Path:
        """Path for an in-progress download; only ever renamed into place
        by `finalize()` after validation succeeds -- never read as final."""

        final = self.get_path(source=source, source_id=source_id, version=version)
        return final.with_name(final.name + self._TEMP_SUFFIX)

    def exists(self, *, source: str, source_id: str, version: str) -> bool:
        return self.get_path(source=source, source_id=source_id, version=version).is_file()

    def finalize(self, temp_path: Path, *, source: str, source_id: str, version: str) -> Path:
        """Atomically move a completed, validated temp download into place.

        `Path.replace` uses `os.replace`, which is atomic on both POSIX and
        Windows (unlike `os.rename`, which fails on Windows if the
        destination exists) -- a reader can never observe a partially
        written file at the final path.
        """

        final_path = self.get_path(source=source, source_id=source_id, version=version)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.replace(final_path)
        return final_path

    def open(self, *, source: str, source_id: str, version: str):
        path = self.get_path(source=source, source_id=source_id, version=version)
        return path.open("rb")

    def delete(self, *, source: str, source_id: str, version: str) -> None:
        self.get_path(source=source, source_id=source_id, version=version).unlink(missing_ok=True)

    def cleanup_temp(self, temp_path: Path) -> None:
        temp_path.unlink(missing_ok=True)
