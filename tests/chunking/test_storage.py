"""Unit tests for `ChunkArtifactStorage`. Pure filesystem, no DB, no chunker."""

import json

from app.core.config import Settings
from app.ingestion.chunking.models import ChunkDiagnostics, ChunkedPaperDocument, ChunkingConfig
from app.ingestion.chunking.storage import ChunkArtifactStorage


def _storage(tmp_path) -> ChunkArtifactStorage:
    return ChunkArtifactStorage(Settings(PAPER_STORAGE_PATH=str(tmp_path)))


def _document(**overrides) -> ChunkedPaperDocument:
    defaults = dict(
        paper_id="paper:arxiv:2401.12345",
        paper_version_id="paper-version:arxiv:2401.12345:v1",
        source_pdf_checksum="pdf-checksum",
        parsed_artifact_checksum="parsed-checksum",
        chunking=ChunkingConfig(
            version="v1", chunk_size_tokens=700, chunk_overlap_tokens=100,
            min_chunk_tokens=80, tokenizer="whitespace-v1", tokenizer_version=None,
            config_fingerprint="fp-a",
        ),
        chunks=[],
        diagnostics=ChunkDiagnostics(
            chunk_count=0, min_tokens=0, max_tokens=0, average_tokens=0,
            median_tokens=0, small_chunk_count=0, oversized_chunk_count=0,
        ),
        warnings=[],
    )
    defaults.update(overrides)
    return ChunkedPaperDocument(**defaults)


class TestDeterministicPaths:
    def test_path_alongside_pdf_and_parsed_json(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        path = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        assert path.name == "chunks.json"
        assert path.parent.name == "v1"

    def test_temp_path_has_part_suffix(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        final = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        temp = storage.get_temp_path(source="arxiv", source_id="2401.12345", version="v1")
        assert temp.name == "chunks.json.part"
        assert temp.parent == final.parent


class TestAtomicWrite:
    def test_write_produces_a_readable_final_file_and_no_temp_file(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        document = _document()

        final_path = storage.write(document, source="arxiv", source_id="2401.12345", version="v1")

        assert final_path.is_file()
        temp_path = storage.get_temp_path(source="arxiv", source_id="2401.12345", version="v1")
        assert not temp_path.exists()

    def test_written_document_round_trips_through_try_read(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        document = _document()
        storage.write(document, source="arxiv", source_id="2401.12345", version="v1")

        read_back = storage.try_read(source="arxiv", source_id="2401.12345", version="v1")

        assert read_back is not None
        assert read_back.paper_id == document.paper_id
        assert read_back.chunking.version == document.chunking.version
        assert read_back.source_pdf_checksum == document.source_pdf_checksum
        assert read_back.parsed_artifact_checksum == document.parsed_artifact_checksum


class TestTryReadMissingOrCorrupt:
    def test_missing_file_returns_none(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        assert storage.try_read(source="arxiv", source_id="2401.12345", version="v1") is None

    def test_malformed_json_returns_none(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        path = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not valid json", encoding="utf-8")

        assert storage.try_read(source="arxiv", source_id="2401.12345", version="v1") is None

    def test_wrong_schema_returns_none(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        path = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"unexpected": true}', encoding="utf-8")

        assert storage.try_read(source="arxiv", source_id="2401.12345", version="v1") is None

    def test_legacy_artifact_without_config_fingerprint_returns_none(self, tmp_path) -> None:
        """Prompt 6.1 backward compatibility (prompt #9): a `chunks.json`
        written before `config_fingerprint` existed must be treated as
        stale/invalid, not silently accepted -- `config_fingerprint` has
        no default, so this is required schema-mismatch behavior, not a
        special-cased check."""

        storage = _storage(tmp_path)
        path = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        path.parent.mkdir(parents=True, exist_ok=True)
        legacy_payload = {
            "paper_id": "paper:arxiv:2401.12345",
            "paper_version_id": "paper-version:arxiv:2401.12345:v1",
            "source_pdf_checksum": "pdf-checksum",
            "parsed_artifact_checksum": "parsed-checksum",
            "chunking": {
                "version": "v1",
                "chunk_size_tokens": 700,
                "chunk_overlap_tokens": 100,
                "min_chunk_tokens": 80,
                "tokenizer": "whitespace-v1",
                # deliberately no "config_fingerprint" key -- the pre-6.1 shape
            },
            "chunks": [],
            "diagnostics": {
                "chunk_count": 0, "min_tokens": 0, "max_tokens": 0, "average_tokens": 0,
                "median_tokens": 0, "small_chunk_count": 0, "oversized_chunk_count": 0,
            },
            "warnings": [],
        }
        path.write_text(json.dumps(legacy_payload), encoding="utf-8")

        assert storage.try_read(source="arxiv", source_id="2401.12345", version="v1") is None


class TestExistenceAndCleanup:
    def test_exists_false_then_true_after_write(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        assert storage.exists(source="arxiv", source_id="2401.12345", version="v1") is False
        storage.write(_document(), source="arxiv", source_id="2401.12345", version="v1")
        assert storage.exists(source="arxiv", source_id="2401.12345", version="v1") is True

    def test_delete_removes_the_artifact(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        storage.write(_document(), source="arxiv", source_id="2401.12345", version="v1")
        storage.delete(source="arxiv", source_id="2401.12345", version="v1")
        assert storage.exists(source="arxiv", source_id="2401.12345", version="v1") is False

    def test_cleanup_temp_is_safe_when_nothing_exists(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        temp_path = storage.get_temp_path(source="arxiv", source_id="2401.12345", version="v1")
        storage.cleanup_temp(temp_path)  # must not raise
