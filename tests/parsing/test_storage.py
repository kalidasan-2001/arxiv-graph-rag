"""Unit tests for `ParsedArtifactStorage`. Pure filesystem, no DB, no parser."""

from app.core.config import Settings
from app.ingestion.parsing.models import ParsedPage, ParsedPaperDocument
from app.ingestion.parsing.storage import ParsedArtifactStorage


def _storage(tmp_path) -> ParsedArtifactStorage:
    return ParsedArtifactStorage(Settings(PAPER_STORAGE_PATH=str(tmp_path)))


def _document(**overrides) -> ParsedPaperDocument:
    defaults = dict(
        paper_id="paper:arxiv:2401.12345",
        paper_version_id="paper-version:arxiv:2401.12345:v1",
        pages=[ParsedPage(page_number=1, text="Hello.")],
        full_text="Hello.",
        sections=[],
        parser_name="pymupdf",
        parser_version="1.28.2+adapter1",
        source_pdf_checksum="abc123",
        page_count=1,
        warnings=[],
    )
    defaults.update(overrides)
    return ParsedPaperDocument(**defaults)


class TestDeterministicPaths:
    def test_path_alongside_pdf_directory(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        path = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        assert path.name == "parsed.json"
        assert path.parent.name == "v1"
        assert path.parent.parent.name == "2401.12345"

    def test_temp_path_has_part_suffix(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        final = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        temp = storage.get_temp_path(source="arxiv", source_id="2401.12345", version="v1")
        assert temp.name == "parsed.json.part"
        assert temp.parent == final.parent


class TestAtomicWrite:
    def test_write_produces_a_readable_final_file(self, tmp_path) -> None:
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
        assert read_back.parser_name == document.parser_name
        assert read_back.parser_version == document.parser_version
        assert read_back.source_pdf_checksum == document.source_pdf_checksum

    def test_write_serializes_sections_correctly(self, tmp_path) -> None:
        from app.domain.enums import SectionType
        from app.domain.papers import PaperSection

        storage = _storage(tmp_path)
        section = PaperSection.create(
            paper_id="paper:arxiv:2401.12345",
            paper_version_id="paper-version:arxiv:2401.12345:v1",
            section_type=SectionType.ABSTRACT,
            order=0,
            text="An abstract.",
            page_start=1,
            page_end=1,
        )
        document = _document(sections=[section])
        storage.write(document, source="arxiv", source_id="2401.12345", version="v1")

        read_back = storage.try_read(source="arxiv", source_id="2401.12345", version="v1")
        assert len(read_back.sections) == 1
        assert read_back.sections[0].section_type == SectionType.ABSTRACT


class TestTryReadMissingOrCorrupt:
    def test_missing_file_returns_none(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        assert storage.try_read(source="arxiv", source_id="2401.12345", version="v1") is None

    def test_malformed_json_returns_none(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        path = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not valid json", encoding="utf-8")

        assert storage.try_read(source="arxiv", source_id="2401.12345", version="v1") is None

    def test_valid_json_with_wrong_schema_returns_none(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        path = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"unexpected": "shape"}', encoding="utf-8")

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
