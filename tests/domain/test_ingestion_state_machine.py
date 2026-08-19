"""Unit tests for the ingestion state machine and resume-point calculation.

Pure-Python, no database -- these must always run, independent of whether
PostgreSQL is available.
"""

import pytest

from app.core.exceptions import InvalidIngestionTransitionError
from app.domain.enums import IngestionStatus, ProcessingStage
from app.domain.ingestion import (
    TERMINAL_STATUSES,
    can_transition,
    get_resume_point,
    validate_transition,
)

_LINEAR_HAPPY_PATH = [
    IngestionStatus.DISCOVERED,
    IngestionStatus.DOWNLOADING,
    IngestionStatus.DOWNLOADED,
    IngestionStatus.PARSING,
    IngestionStatus.PARSED,
    IngestionStatus.CHUNKING,
    IngestionStatus.CHUNKED,
    IngestionStatus.VECTOR_INDEXING,
    IngestionStatus.VECTOR_INDEXED,
    IngestionStatus.GRAPH_INDEXING,
    IngestionStatus.GRAPH_INDEXED,
    IngestionStatus.READY,
]


class TestValidTransitions:
    @pytest.mark.parametrize(
        ("current", "target"),
        list(zip(_LINEAR_HAPPY_PATH, _LINEAR_HAPPY_PATH[1:])),
    )
    def test_linear_happy_path_step_is_allowed(
        self, current: IngestionStatus, target: IngestionStatus
    ) -> None:
        assert can_transition(current, target)
        validate_transition(current, target)  # must not raise

    @pytest.mark.parametrize(
        "active_status",
        [s for s in IngestionStatus if s not in TERMINAL_STATUSES],
    )
    def test_any_active_status_can_fail(self, active_status: IngestionStatus) -> None:
        assert can_transition(active_status, IngestionStatus.FAILED)


class TestInvalidTransitions:
    def test_cannot_skip_directly_from_discovered_to_ready(self) -> None:
        assert not can_transition(IngestionStatus.DISCOVERED, IngestionStatus.READY)
        with pytest.raises(InvalidIngestionTransitionError):
            validate_transition(IngestionStatus.DISCOVERED, IngestionStatus.READY)

    def test_cannot_go_backwards(self) -> None:
        assert not can_transition(IngestionStatus.PARSED, IngestionStatus.DOWNLOADING)

    def test_ready_is_terminal(self) -> None:
        for target in IngestionStatus:
            assert not can_transition(IngestionStatus.READY, target)

    def test_failed_is_terminal_in_this_stage(self) -> None:
        # No automatic-retry / recovery transition is implemented yet
        # (prompt #18/#38) -- FAILED has no legal outgoing transition here.
        for target in IngestionStatus:
            assert not can_transition(IngestionStatus.FAILED, target)

    def test_cannot_transition_to_self(self) -> None:
        assert not can_transition(IngestionStatus.PARSING, IngestionStatus.PARSING)


class TestResumePoint:
    def test_downloaded_resumes_at_parse(self) -> None:
        assert get_resume_point(IngestionStatus.DOWNLOADED) == ProcessingStage.PARSE

    def test_parsed_resumes_at_chunk(self) -> None:
        assert get_resume_point(IngestionStatus.PARSED) == ProcessingStage.CHUNK

    def test_vector_indexed_resumes_at_graph_index(self) -> None:
        assert get_resume_point(IngestionStatus.VECTOR_INDEXED) == ProcessingStage.GRAPH_INDEX

    def test_discovered_resumes_at_download(self) -> None:
        assert get_resume_point(IngestionStatus.DISCOVERED) == ProcessingStage.DOWNLOAD

    def test_in_progress_status_resumes_at_the_same_stage_it_was_attempting(self) -> None:
        # An "-ING" status found on resume means the attempt was
        # interrupted; it must be re-attempted, not skipped.
        assert get_resume_point(IngestionStatus.PARSING) == ProcessingStage.PARSE

    def test_ready_has_no_resume_point(self) -> None:
        assert get_resume_point(IngestionStatus.READY) is None

    def test_graph_indexed_has_no_resume_point(self) -> None:
        # Nothing left to process; the only remaining transition is -> READY.
        assert get_resume_point(IngestionStatus.GRAPH_INDEXED) is None

    def test_failed_resumes_from_its_failed_stage_not_from_failed_itself(self) -> None:
        resume = get_resume_point(IngestionStatus.FAILED, failed_stage=IngestionStatus.PARSED)
        assert resume == ProcessingStage.CHUNK

    def test_failed_without_failed_stage_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            get_resume_point(IngestionStatus.FAILED, failed_stage=None)
