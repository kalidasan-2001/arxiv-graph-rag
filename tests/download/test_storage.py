"""Unit tests for `PaperStorage`. Pure filesystem, no network, no DB."""

import pytest

from app.core.config import Settings
from app.core.exceptions import InvalidStoragePathError
from app.ingestion.download.storage import PaperStorage


def _storage(tmp_path) -> PaperStorage:
    return PaperStorage(Settings(PAPER_STORAGE_PATH=str(tmp_path)))


class TestDeterministicPaths:
    def test_path_is_deterministic_for_the_same_inputs(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        first = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        second = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        assert first == second

    def test_different_versions_get_different_paths_under_the_same_paper(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        v1 = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        v2 = storage.get_path(source="arxiv", source_id="2401.12345", version="v2")
        assert v1 != v2
        assert v1.parent.parent == v2.parent.parent  # same logical-paper directory

    def test_layout_matches_source_source_id_version(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        path = storage.get_path(source="arxiv", source_id="2401.12345", version="v2")
        assert path.name == "paper.pdf"
        assert path.parent.name == "v2"
        assert path.parent.parent.name == "2401.12345"
        assert path.parent.parent.parent.name == "arxiv"

    def test_temp_path_is_alongside_final_with_part_suffix(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        final = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        temp = storage.get_temp_path(source="arxiv", source_id="2401.12345", version="v1")
        assert temp.parent == final.parent
        assert temp.name == "paper.pdf.part"


class TestPathTraversalPrevention:
    @pytest.mark.parametrize("bad_value", ["../escape", "..", ".", "a/b", "a\\b", "", "   "])
    def test_unsafe_source_id_is_rejected(self, tmp_path, bad_value: str) -> None:
        storage = _storage(tmp_path)
        with pytest.raises(InvalidStoragePathError):
            storage.get_path(source="arxiv", source_id=bad_value, version="v1")

    @pytest.mark.parametrize("bad_value", ["../escape", "..", "a/b"])
    def test_unsafe_version_is_rejected(self, tmp_path, bad_value: str) -> None:
        storage = _storage(tmp_path)
        with pytest.raises(InvalidStoragePathError):
            storage.get_path(source="arxiv", source_id="2401.12345", version=bad_value)

    def test_safe_values_are_accepted(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        path = storage.get_path(source="arxiv", source_id="2401.12345", version="v1")
        assert str(storage._root) in str(path)  # noqa: SLF001 -- verifying the safety invariant


class TestAtomicFinalization:
    def test_finalize_moves_temp_file_to_final_path(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        temp_path = storage.get_temp_path(source="arxiv", source_id="2401.12345", version="v1")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(b"%PDF-1.4 fake content")

        final_path = storage.finalize(
            temp_path, source="arxiv", source_id="2401.12345", version="v1"
        )

        assert final_path.is_file()
        assert not temp_path.exists()
        assert final_path.read_bytes() == b"%PDF-1.4 fake content"

    def test_finalize_creates_parent_directories(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        temp_path = storage.get_temp_path(source="arxiv", source_id="2401.99999", version="v3")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(b"%PDF-1.4")

        final_path = storage.finalize(
            temp_path, source="arxiv", source_id="2401.99999", version="v3"
        )
        assert final_path.exists()


class TestTemporaryFileCleanup:
    def test_cleanup_temp_removes_a_partial_file(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        temp_path = storage.get_temp_path(source="arxiv", source_id="2401.12345", version="v1")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(b"partial")

        storage.cleanup_temp(temp_path)

        assert not temp_path.exists()

    def test_cleanup_temp_is_safe_when_nothing_exists(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        temp_path = storage.get_temp_path(source="arxiv", source_id="2401.12345", version="v1")
        storage.cleanup_temp(temp_path)  # must not raise


class TestArtifactExistence:
    def test_exists_is_false_before_finalization(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        assert storage.exists(source="arxiv", source_id="2401.12345", version="v1") is False

    def test_exists_is_true_after_finalization(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        temp_path = storage.get_temp_path(source="arxiv", source_id="2401.12345", version="v1")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(b"%PDF-1.4")
        storage.finalize(temp_path, source="arxiv", source_id="2401.12345", version="v1")

        assert storage.exists(source="arxiv", source_id="2401.12345", version="v1") is True

    def test_delete_removes_the_artifact(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        temp_path = storage.get_temp_path(source="arxiv", source_id="2401.12345", version="v1")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(b"%PDF-1.4")
        storage.finalize(temp_path, source="arxiv", source_id="2401.12345", version="v1")

        storage.delete(source="arxiv", source_id="2401.12345", version="v1")

        assert storage.exists(source="arxiv", source_id="2401.12345", version="v1") is False

    def test_delete_is_safe_when_nothing_exists(self, tmp_path) -> None:
        storage = _storage(tmp_path)
        storage.delete(source="arxiv", source_id="2401.12345", version="v1")  # must not raise
