"""Local filesystem storage for structured parse artifacts (`parsed.json`).

Lives alongside the raw PDF at the same deterministic paper-version
directory `PaperStorage` (Prompt 4) already uses -- just a different
filename::

    {PAPER_STORAGE_PATH}/{source}/{source_id}/{version}/parsed.json

Same atomicity discipline as PDF acquisition (prompt #26): writes go to a
`.part` temp file first, and only an `os.replace` (`Path.replace`) moves a
fully-serialized, complete file into the final name -- a partial crash
during `json.dumps`/write can never leave `parsed.json` looking complete.
"""

import json
from pathlib import Path

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.ingestion.paper_paths import paper_version_directory
from app.ingestion.parsing.models import ParsedPaperDocument


class ParsedArtifactStorage:
    """Filesystem-backed storage for structured `parsed.json` artifacts."""

    _FINAL_NAME = "parsed.json"
    _TEMP_SUFFIX = ".part"

    def __init__(self, settings: Settings) -> None:
        if not settings.PAPER_STORAGE_PATH:
            raise ConfigurationError("PAPER_STORAGE_PATH is not configured")
        self._root = Path(settings.PAPER_STORAGE_PATH).resolve()

    def get_path(self, *, source: str, source_id: str, version: str) -> Path:
        directory = paper_version_directory(
            self._root, source=source, source_id=source_id, version=version
        )
        return directory / self._FINAL_NAME

    def get_temp_path(self, *, source: str, source_id: str, version: str) -> Path:
        final = self.get_path(source=source, source_id=source_id, version=version)
        return final.with_name(final.name + self._TEMP_SUFFIX)

    def exists(self, *, source: str, source_id: str, version: str) -> bool:
        return self.get_path(source=source, source_id=source_id, version=version).is_file()

    def write(
        self, document: ParsedPaperDocument, *, source: str, source_id: str, version: str
    ) -> Path:
        """Serialize `document`, write to a temp file, then atomically finalize."""

        temp_path = self.get_temp_path(source=source, source_id=source_id, version=version)
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        payload = document.model_dump(mode="json")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        final_path = self.get_path(source=source, source_id=source_id, version=version)
        temp_path.replace(final_path)
        return final_path

    def try_read(
        self, *, source: str, source_id: str, version: str
    ) -> ParsedPaperDocument | None:
        """Read and validate `parsed.json`.

        Returns `None` if the file is missing *or* corrupt (malformed
        JSON, schema mismatch) rather than raising -- callers treat both
        cases identically: "no valid parse artifact exists here" (prompt
        #30's reconciliation policy for a malformed artifact).
        """

        path = self.get_path(source=source, source_id=source_id, version=version)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ParsedPaperDocument.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, UnicodeDecodeError, OSError):
            return None

    def delete(self, *, source: str, source_id: str, version: str) -> None:
        self.get_path(source=source, source_id=source_id, version=version).unlink(missing_ok=True)

    def cleanup_temp(self, temp_path: Path) -> None:
        temp_path.unlink(missing_ok=True)
