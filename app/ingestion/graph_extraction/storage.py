"""Local filesystem storage for structured graph extraction artifacts
(`graph_extraction.json`).

Lives alongside `paper.pdf`, `parsed.json`, and `chunks.json` at the same
deterministic paper-version directory (Prompt 4/5/6)::

    {PAPER_STORAGE_PATH}/{source}/{source_id}/{version}/graph_extraction.json

Same atomicity discipline as every prior artifact stage: writes go to a
`.part` temp file first, and only an `os.replace` (`Path.replace`) moves a
fully-serialized, complete file into the final name -- a partial crash can
never leave a `graph_extraction.json` that looks complete (prompt #28).
"""

import json
from pathlib import Path

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.ingestion.graph_extraction.models import GraphExtractionArtifact
from app.ingestion.paper_paths import paper_version_directory


class GraphExtractionArtifactStorage:
    """Filesystem-backed storage for structured `graph_extraction.json` artifacts."""

    _FINAL_NAME = "graph_extraction.json"
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
        self, artifact: GraphExtractionArtifact, *, source: str, source_id: str, version: str
    ) -> Path:
        """Serialize `artifact`, write to a temp file, then atomically finalize."""

        temp_path = self.get_temp_path(source=source, source_id=source_id, version=version)
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        payload = artifact.model_dump(mode="json")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        final_path = self.get_path(source=source, source_id=source_id, version=version)
        temp_path.replace(final_path)
        return final_path

    def try_read(
        self, *, source: str, source_id: str, version: str
    ) -> GraphExtractionArtifact | None:
        """Read and validate `graph_extraction.json`.

        Returns `None` if missing *or* corrupt (malformed JSON, schema
        mismatch) rather than raising -- callers treat both cases
        identically: "no valid extraction artifact exists here."
        """

        path = self.get_path(source=source, source_id=source_id, version=version)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return GraphExtractionArtifact.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, UnicodeDecodeError, OSError):
            return None

    def delete(self, *, source: str, source_id: str, version: str) -> None:
        self.get_path(source=source, source_id=source_id, version=version).unlink(missing_ok=True)

    def cleanup_temp(self, temp_path: Path) -> None:
        temp_path.unlink(missing_ok=True)
