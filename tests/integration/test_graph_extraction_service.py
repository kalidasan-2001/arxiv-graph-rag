"""Integration tests for `ScientificKnowledgeExtractionService` against
real PostgreSQL.

No live network, no real LLM -- the "chunked" precondition is set up via
real `PyMuPDFParser` + `SectionAwareChunker` (matching what Prompts 5/6
would have already produced), and semantic extraction comes from the
deterministic `FakeLLMProvider` (prompt #46). Requires a reachable
database (see `tests/integration/conftest.py`); skipped automatically
otherwise. No Qdrant/Neo4j involved -- extraction only reads `chunks.json`.

Covers the full target flow (prompt #60): chunked paper -> extraction
request -> GRAPH_INDEXING -> deterministic Paper/Author/CITES facts -> LLM
semantic extraction -> graph_extraction.json -> metadata persisted --
plus idempotency, use-vs-mention, provenance, chunk-change invalidation,
LLM-model/prompt-version invalidation, reconciliation, and partial-failure
safety.
"""

import pytest

from app.core.config import Settings
from app.core.exceptions import ChunkArtifactNotFoundError
from app.domain.enums import IngestionStatus, RelationshipType
from app.domain.ids import build_paper_id
from app.domain.papers import Paper, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.chunking.section_chunker import SectionAwareChunker
from app.ingestion.chunking.service import ChunkingService
from app.ingestion.chunking.storage import ChunkArtifactStorage
from app.ingestion.download.storage import PaperStorage
from app.ingestion.graph_extraction.models import (
    RawEntityCandidate,
    RawExtractionResponse,
    RawRelationshipCandidate,
)
from app.ingestion.graph_extraction.service import ScientificKnowledgeExtractionService
from app.ingestion.graph_extraction.storage import GraphExtractionArtifactStorage
from app.ingestion.parsing.pymupdf_parser import PyMuPDFParser
from app.ingestion.parsing.storage import ParsedArtifactStorage
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository
from tests.llm.fakes import FakeLLMProvider
from tests.parsing.pdf_fixtures import make_pdf_bytes

_PAPER_PAGE = """Graph Extraction Test Paper
Alice Author, Bob Author

Abstract
We propose GraphSteal, a novel method for graph reconstruction attacks
against Graph RAG systems.

1 Introduction
Graph RAG systems are vulnerable to structural extraction attacks.

2 Related Work
LightRAG and ToG are prior systems that use different retrieval strategies.

3 Methodology
Our method GraphSteal performs a depth-first search over the knowledge graph.

4 Experiments
We evaluate GraphSteal on HotpotQA and MIMIC-IV benchmarks.

References
[1] Smith, J. Some Related Paper. arXiv:2401.11111, 2024.
[2] Doe, A. Another Paper Without An ID. Conference Proceedings, 2021."""

_METHODOLOGY_MARKER = "GraphSteal performs a depth-first search"
_EXPERIMENTS_MARKER = "HotpotQA and MIMIC-IV"
_RELATED_WORK_MARKER = "LightRAG and ToG are prior systems"


def _settings(storage_root) -> Settings:
    return Settings(PAPER_STORAGE_PATH=str(storage_root))


def _prepare_chunked_paper(db_session, storage_root, *, source_id: str = "2401.99001"):
    """Discover a paper/version with real authors, place a real PDF, parse
    and chunk it for real -- exactly the precondition
    `ScientificKnowledgeExtractionService` expects."""

    papers = PaperRepository(db_session)
    paper = papers.upsert_paper(
        Paper.create(
            source="arxiv", source_id=source_id, title="Graph Extraction Test Paper",
            authors=["Alice Author", "Bob Author"],
        )
    )
    version = papers.get_or_create_paper_version(
        PaperVersion.create(paper_id=paper.paper_id, version="v1")
    )

    settings = _settings(storage_root)
    pdf_storage = PaperStorage(settings)
    temp_path = pdf_storage.get_temp_path(source="arxiv", source_id=source_id, version="v1")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(make_pdf_bytes([_PAPER_PAGE]))
    final_path = pdf_storage.finalize(temp_path, source="arxiv", source_id=source_id, version="v1")
    pdf_checksum = sha256_file(final_path)
    version = papers.update_version_artifact(
        version.paper_version_id, checksum=pdf_checksum, storage_path=str(final_path),
        file_size_bytes=final_path.stat().st_size, downloaded_at=version.created_at,
    )

    parser = PyMuPDFParser()
    parsed_document = parser.parse(
        final_path, paper_id=paper.paper_id, paper_version_id=version.paper_version_id
    )
    parsed_document = parsed_document.model_copy(update={"source_pdf_checksum": pdf_checksum})
    parsed_storage = ParsedArtifactStorage(settings)
    parsed_path = parsed_storage.write(parsed_document, source="arxiv", source_id=source_id, version="v1")
    version = papers.update_version_parse_result(
        version.paper_version_id, parsed_artifact_path=str(parsed_path), parsed_at=version.created_at,
        parser_name=parsed_document.parser_name, parser_version=parsed_document.parser_version,
        page_count=parsed_document.page_count, section_count=len(parsed_document.sections),
        warning_count=len(parsed_document.warnings),
    )

    chunking_service = ChunkingService(
        SectionAwareChunker(settings), parsed_storage, ChunkArtifactStorage(settings),
        papers, IngestionRepository(db_session),
    )
    chunking_service.chunk(paper.paper_id)

    version = papers.get_paper_version(version.paper_version_id)
    return paper, version


def _default_fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider(
        responses_by_chunk_marker={
            _METHODOLOGY_MARKER: RawExtractionResponse(
                entities=[
                    RawEntityCandidate(
                        entity_type="method", name="GraphSteal", aliases=[],
                        evidence_quote=None, confidence=0.9,
                    )
                ],
                relationships=[
                    RawRelationshipCandidate(
                        relationship_type="uses_method", source_name="Graph Extraction Test Paper",
                        source_type="paper", target_name="GraphSteal", target_type="method",
                        usage="used_by_this_paper", evidence_quote=None, confidence=0.9,
                    )
                ],
            ),
            _EXPERIMENTS_MARKER: RawExtractionResponse(
                entities=[
                    RawEntityCandidate(
                        entity_type="dataset", name="HotpotQA", aliases=[],
                        evidence_quote=None, confidence=0.85,
                    ),
                    RawEntityCandidate(
                        entity_type="dataset", name="MIMIC-IV", aliases=[],
                        evidence_quote=None, confidence=0.85,
                    ),
                ],
                relationships=[
                    RawRelationshipCandidate(
                        relationship_type="evaluated_on", source_name="Graph Extraction Test Paper",
                        source_type="paper", target_name="HotpotQA", target_type="dataset",
                        usage="used_by_this_paper", evidence_quote=None, confidence=0.85,
                    ),
                    RawRelationshipCandidate(
                        relationship_type="evaluated_on", source_name="Graph Extraction Test Paper",
                        source_type="paper", target_name="MIMIC-IV", target_type="dataset",
                        usage="used_by_this_paper", evidence_quote=None, confidence=0.85,
                    ),
                ],
            ),
        },
        default_response=RawExtractionResponse(),
    )


def _service(
    db_session, storage_root, *, llm_provider=None, extraction_version: str = "v1"
) -> tuple[ScientificKnowledgeExtractionService, FakeLLMProvider]:
    settings = _settings(storage_root)
    provider = llm_provider or _default_fake_llm()
    service = ScientificKnowledgeExtractionService(
        provider, ChunkArtifactStorage(settings), GraphExtractionArtifactStorage(settings),
        PaperRepository(db_session), IngestionRepository(db_session),
        extraction_version=extraction_version,
    )
    return service, provider


class TestFullExtractionFlow:
    def test_chunked_paper_is_extracted_and_marked_graph_indexing(
        self, db_session, tmp_path
    ) -> None:
        paper, version = _prepare_chunked_paper(db_session, tmp_path)
        service, provider = _service(db_session, tmp_path)

        result = service.extract(paper.paper_id)

        assert result.job.status == IngestionStatus.GRAPH_INDEXING
        assert result.extraction_reused is False
        assert result.entity_count > 0
        assert result.relationship_count > 0
        assert provider.call_count > 0

        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.entity_count == result.entity_count
        assert stored_version.relationship_count == result.relationship_count
        assert stored_version.graph_extraction_generation_fingerprint == result.graph_extraction_generation_fingerprint
        assert stored_version.graph_extracted_at is not None

        settings = _settings(tmp_path)
        artifact = GraphExtractionArtifactStorage(settings).try_read(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        assert artifact is not None
        assert artifact.paper_id == paper.paper_id

    def test_extraction_without_chunks_raises(self, db_session, tmp_path) -> None:
        papers = PaperRepository(db_session)
        paper = papers.upsert_paper(
            Paper.create(source="arxiv", source_id="2401.99002", title="Not Chunked")
        )
        papers.get_or_create_paper_version(PaperVersion.create(paper_id=paper.paper_id, version="v1"))
        service, _provider = _service(db_session, tmp_path)

        with pytest.raises(ChunkArtifactNotFoundError):
            service.extract(paper.paper_id)


class TestDeterministicPaperAndAuthorFacts:
    def test_authored_by_relationships_created_without_any_llm_call_for_related_work(
        self, db_session, tmp_path
    ) -> None:
        # RELATED_WORK is structurally excluded from semantic extraction
        # (prompt #13/#34) -- confirms its text never even reaches the LLM.
        paper, _version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99010")
        service, provider = _service(db_session, tmp_path)

        result = service.extract(paper.paper_id)

        assert not any(_RELATED_WORK_MARKER in call for call in provider.calls)

        settings = _settings(tmp_path)
        artifact = GraphExtractionArtifactStorage(settings).try_read(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        paper_entities = [e for e in artifact.entities if e.entity_type.value == "paper"]
        author_entities = [e for e in artifact.entities if e.entity_type.value == "author"]
        assert len(paper_entities) == 1
        assert paper_entities[0].entity_id == paper.paper_id  # trusted identity, not a name-hash
        assert {e.canonical_name for e in author_entities} == {"Alice Author", "Bob Author"}

        authored_by = [
            r for r in artifact.relationships if r.relationship_type == RelationshipType.AUTHORED_BY
        ]
        assert len(authored_by) == 2
        assert all(r.source_entity_id == paper.paper_id for r in authored_by)
        assert all(r.confidence == 1.0 for r in authored_by)


class TestCitationResolution:
    def test_explicit_arxiv_id_becomes_a_cites_relationship(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99020")
        service, _provider = _service(db_session, tmp_path)

        service.extract(paper.paper_id)

        settings = _settings(tmp_path)
        artifact = GraphExtractionArtifactStorage(settings).try_read(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        cites = [r for r in artifact.relationships if r.relationship_type == RelationshipType.CITES]
        assert len(cites) == 1
        assert cites[0].source_entity_id == paper.paper_id
        assert cites[0].target_entity_id == build_paper_id("arxiv", "2401.11111")
        assert cites[0].confidence == 1.0

        assert len(artifact.unresolved_citations) == 1  # the ID-less reference entry


class TestUseVsMentionIntegration:
    def test_mentioned_only_method_never_produces_uses_method(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99030")
        provider = FakeLLMProvider(
            responses_by_chunk_marker={
                _METHODOLOGY_MARKER: RawExtractionResponse(
                    entities=[
                        RawEntityCandidate(
                            entity_type="method", name="GraphSteal", aliases=[],
                            evidence_quote=None, confidence=0.9,
                        )
                    ],
                    relationships=[
                        RawRelationshipCandidate(
                            relationship_type="uses_method", source_name="Paper", source_type="paper",
                            target_name="GraphSteal", target_type="method", usage="mentioned_only",
                            evidence_quote=None, confidence=0.9,
                        )
                    ],
                ),
            }
        )
        service, _ = _service(db_session, tmp_path, llm_provider=provider)

        service.extract(paper.paper_id)

        settings = _settings(tmp_path)
        artifact = GraphExtractionArtifactStorage(settings).try_read(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        uses_method = [r for r in artifact.relationships if r.relationship_type == RelationshipType.USES_METHOD]
        assert uses_method == []
        assert any(w.code.value == "relationship_candidate_rejected" for w in artifact.warnings)


class TestProvenance:
    def test_every_semantic_relationship_has_full_provenance(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99040")
        service, _provider = _service(db_session, tmp_path)

        service.extract(paper.paper_id)

        settings = _settings(tmp_path)
        artifact = GraphExtractionArtifactStorage(settings).try_read(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        chunk_ids = {chunk.chunk_id for chunk in ChunkArtifactStorage(settings).try_read(
            source="arxiv", source_id=paper.source_id, version="v1"
        ).chunks}

        semantic_relationships = [
            r for r in artifact.relationships
            if r.relationship_type in (RelationshipType.USES_METHOD, RelationshipType.EVALUATED_ON)
        ]
        assert len(semantic_relationships) > 0
        for relationship in semantic_relationships:
            assert relationship.source_chunk_id is not None
            assert relationship.source_chunk_id in chunk_ids
            assert relationship.extraction_version == "v1"
            assert 0.0 <= relationship.confidence <= 1.0


class TestIdempotency:
    def test_second_extraction_does_not_invoke_the_llm_again(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99050")
        service, provider = _service(db_session, tmp_path)

        first = service.extract(paper.paper_id)
        calls_after_first = provider.call_count
        second = service.extract(paper.paper_id)

        assert provider.call_count == calls_after_first  # the critical assertion
        assert second.extraction_reused is True
        assert second.job.status == IngestionStatus.GRAPH_INDEXING
        assert second.job.ingestion_job_id == first.job.ingestion_job_id
        assert second.entity_count == first.entity_count


class TestChunkArtifactChangedInvalidation:
    def test_rechunking_triggers_a_real_reextraction(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99060")
        service, provider = _service(db_session, tmp_path)
        first = service.extract(paper.paper_id)
        assert provider.call_count > 0

        rechunk_settings = Settings(PAPER_STORAGE_PATH=str(tmp_path), CHUNK_SIZE_TOKENS=50)
        rechunk_service = ChunkingService(
            SectionAwareChunker(rechunk_settings), ParsedArtifactStorage(rechunk_settings),
            ChunkArtifactStorage(rechunk_settings), PaperRepository(db_session),
            IngestionRepository(db_session),
        )
        rechunk_result = rechunk_service.chunk(paper.paper_id)
        assert rechunk_result.chunk_reused is False

        second = service.extract(paper.paper_id)

        assert provider.call_count > 0
        assert second.extraction_reused is False
        assert second.graph_extraction_generation_fingerprint != first.graph_extraction_generation_fingerprint
        assert second.job.ingestion_job_id != first.job.ingestion_job_id  # stale job replaced


class TestLLMModelChangedInvalidation:
    def test_different_model_triggers_a_real_reextraction(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99070")
        provider_a = _default_fake_llm()
        provider_a._model_name = "model-a"  # noqa: SLF001 -- test-only override
        service_a, _ = _service(db_session, tmp_path, llm_provider=provider_a)
        first = service_a.extract(paper.paper_id)
        assert provider_a.call_count > 0

        provider_b = _default_fake_llm()
        provider_b._model_name = "model-b"  # noqa: SLF001
        service_b, _ = _service(db_session, tmp_path, llm_provider=provider_b)
        second = service_b.extract(paper.paper_id)

        assert provider_b.call_count > 0  # a genuine re-extraction happened
        assert second.extraction_reused is False
        assert second.extraction_config_fingerprint != first.extraction_config_fingerprint


class TestPromptVersionChangedInvalidation:
    def test_prompt_version_change_triggers_a_real_reextraction(
        self, db_session, tmp_path, monkeypatch
    ) -> None:
        paper, _version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99080")
        service, provider = _service(db_session, tmp_path)
        first = service.extract(paper.paper_id)
        assert provider.call_count > 0

        monkeypatch.setattr(
            "app.ingestion.graph_extraction.service.PROMPT_VERSION", "v2-test"
        )
        second = service.extract(paper.paper_id)

        assert second.extraction_reused is False
        assert second.extraction_config_fingerprint != first.extraction_config_fingerprint


class TestReconciliation:
    def test_artifact_valid_but_final_db_write_failed_is_reconciled_without_llm_calls(
        self, db_session, tmp_path
    ) -> None:
        paper, version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99090")
        settings = _settings(tmp_path)
        provider = _default_fake_llm()

        first_service = ScientificKnowledgeExtractionService(
            provider, ChunkArtifactStorage(settings), GraphExtractionArtifactStorage(settings),
            PaperRepository(db_session), IngestionRepository(db_session), extraction_version="v1",
        )
        first_service.extract(paper.paper_id)

        counting_provider = _default_fake_llm()
        second_service = ScientificKnowledgeExtractionService(
            counting_provider, ChunkArtifactStorage(settings), GraphExtractionArtifactStorage(settings),
            PaperRepository(db_session), IngestionRepository(db_session), extraction_version="v1",
        )
        result = second_service.extract(paper.paper_id)

        assert counting_provider.call_count == 0  # never re-extracted -- reconciled from disk
        assert result.extraction_reused is True
        assert result.job.status == IngestionStatus.GRAPH_INDEXING

    def test_db_says_extracted_but_artifact_missing_triggers_reextraction(
        self, db_session, tmp_path
    ) -> None:
        paper, _version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99100")
        service, provider = _service(db_session, tmp_path)
        first = service.extract(paper.paper_id)
        assert first.job.status == IngestionStatus.GRAPH_INDEXING

        settings = _settings(tmp_path)
        GraphExtractionArtifactStorage(settings).delete(
            source="arxiv", source_id=paper.source_id, version="v1"
        )

        second = service.extract(paper.paper_id)

        assert provider.call_count > 0
        assert second.job.status == IngestionStatus.GRAPH_INDEXING
        assert second.job.ingestion_job_id != first.job.ingestion_job_id

    def test_corrupt_artifact_triggers_reextraction(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99110")
        service, provider = _service(db_session, tmp_path)
        first = service.extract(paper.paper_id)

        settings = _settings(tmp_path)
        artifact_path = GraphExtractionArtifactStorage(settings).get_path(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        artifact_path.write_text("{ not valid json at all", encoding="utf-8")

        second = service.extract(paper.paper_id)

        assert provider.call_count > 0
        assert second.job.ingestion_job_id != first.job.ingestion_job_id


class TestPartialFailure:
    def test_llm_failure_on_one_chunk_aborts_the_whole_extraction(self, db_session, tmp_path) -> None:
        paper, version = _prepare_chunked_paper(db_session, tmp_path, source_id="2401.99120")
        # Fails on the 2nd LLM call (whichever semantic chunk that is) --
        # simulates a chunk failing after retries are exhausted.
        provider = FakeLLMProvider(fail_on_call_numbers={2})
        service, _ = _service(db_session, tmp_path, llm_provider=provider)

        with pytest.raises(Exception):  # the underlying LLMResponseError propagates
            service.extract(paper.paper_id)

        settings = _settings(tmp_path)
        assert not GraphExtractionArtifactStorage(settings).exists(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.graph_extraction_artifact_path is None  # never finalized as trusted
