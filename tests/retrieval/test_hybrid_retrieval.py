import pytest

from app.core.exceptions import HybridRetrievalError, VectorSearchError
from app.domain.enums import (
    EntityType,
    EvidenceScoreKind,
    EvidenceSourceStore,
    EvidenceType,
    RetrievalStrategy,
)
from app.domain.evidence import EvidenceItem, EvidenceProvenance
from app.domain.ids import build_evidence_id
from app.retrieval.evidence import EvidenceBridgeResult, build_text_evidence_id
from app.retrieval.graph_search import GraphSearchOperation
from app.retrieval.hybrid import EvidenceFusionService, HybridRetrievalService
from app.storage.qdrant.models import VectorSearchHit


def _text(
    chunk_id: str,
    *,
    score: float = 0.8,
    generation: str = "vector-current",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=build_text_evidence_id(chunk_id, generation),
        evidence_type=EvidenceType.TEXT,
        paper_id="paper:arxiv:a",
        paper_version_id="paper-version:arxiv:a:v1",
        chunk_id=chunk_id,
        section_id="section:intro",
        section_type="introduction",
        page_start=1,
        page_end=1,
        text=f"Text for {chunk_id}",
        score=score,
        score_kind=EvidenceScoreKind.VECTOR_SIMILARITY,
        source="qdrant",
        source_store=EvidenceSourceStore.QDRANT,
        provenance=EvidenceProvenance(
            provenance_type="chunk",
            source_store=EvidenceSourceStore.QDRANT,
            paper_id="paper:arxiv:a",
            paper_version_id="paper-version:arxiv:a:v1",
            chunk_ids=[chunk_id],
            vector_generation_fingerprint=generation,
        ),
    )


def _hit(chunk_id: str, *, score: float = 0.8) -> VectorSearchHit:
    return VectorSearchHit(
        chunk_id=chunk_id,
        paper_id="paper:arxiv:a",
        paper_version_id="paper-version:arxiv:a:v1",
        section_id="section:intro",
        section_type="introduction",
        section_title="Introduction",
        chunk_index=0,
        page_start=1,
        page_end=1,
        text=f"Text for {chunk_id}",
        vector_generation_fingerprint="vector-current",
        similarity_score=score,
    )


def _graph(evidence_id: str, *, support_text_ids: list[str] | None = None) -> EvidenceItem:
    support_text_ids = support_text_ids or []
    return EvidenceItem(
        evidence_id=evidence_id,
        evidence_type=EvidenceType.GRAPH_RELATIONSHIP,
        paper_id="paper:arxiv:a",
        paper_version_id="paper-version:arxiv:a:v1",
        entity_ids=["paper:arxiv:a", "entity:method:m1"],
        relationship_ids=[evidence_id.replace("evidence:", "rel:")],
        source_chunk_ids=["chunk:support"],
        supporting_text_evidence_ids=support_text_ids,
        text="Paper A uses M1.",
        score=0.9,
        score_kind=EvidenceScoreKind.GRAPH_PATH_CONFIDENCE,
        source="neo4j",
        source_store=EvidenceSourceStore.NEO4J,
        provenance=EvidenceProvenance(
            provenance_type="chunk",
            source_store=EvidenceSourceStore.NEO4J,
            paper_id="paper:arxiv:a",
            paper_version_id="paper-version:arxiv:a:v1",
            chunk_ids=["chunk:support"],
            relationship_ids=[evidence_id.replace("evidence:", "rel:")],
            graph_index_generation_fingerprint="graph-current",
            extraction_version="extract-v1",
        ),
    )


class _VectorService:
    def __init__(self, hits=None, error: Exception | None = None) -> None:
        self.hits = hits or []
        self.error = error
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error:
            raise self.error
        return self.hits


class _GraphService:
    def __init__(self, evidence=None, error: Exception | None = None) -> None:
        self.evidence = evidence or []
        self.error = error
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return type("GraphResult", (), {"evidence": self.evidence})()


class _Bridge:
    def __init__(self, support=None) -> None:
        self.support = support or {}
        self.calls = []

    def resolve_graph_evidence_sources(self, evidence):
        self.calls.append(evidence.evidence_id)
        return EvidenceBridgeResult(
            graph_evidence=evidence,
            text_evidence=self.support.get(evidence.evidence_id, []),
        )


def _service(vector_service, graph_service, bridge) -> HybridRetrievalService:
    return HybridRetrievalService(
        vector_service=vector_service,
        graph_service=graph_service,
        provenance_bridge=bridge,
        fusion_service=EvidenceFusionService(rrf_k=60),
        default_top_k=10,
        max_top_k=50,
    )


class TestEvidenceFusionService:
    def test_rrf_formula_and_order_are_deterministic(self) -> None:
        a = _text("chunk:a")
        b = _text("chunk:b")
        c = _text("chunk:c")
        d = _graph(build_evidence_id(EvidenceType.GRAPH_RELATIONSHIP, "d"))

        fused, diagnostics = EvidenceFusionService(rrf_k=60).fuse(
            vector_evidence=[a, b, c],
            graph_evidence=[b, d, a],
            graph_support_evidence=[],
            top_k=10,
        )

        by_id = {item.evidence.evidence_id: item for item in fused}
        assert by_id[b.evidence_id].fusion_score == pytest.approx(1 / 62 + 1 / 61)
        assert by_id[a.evidence_id].fusion_score == pytest.approx(1 / 61 + 1 / 63)
        assert [item.evidence.evidence_id for item in fused][:4] == [
            b.evidence_id,
            a.evidence_id,
            d.evidence_id,
            c.evidence_id,
        ]
        assert diagnostics["duplicate_evidence_removed"] == 2

    def test_text_and_graph_evidence_are_not_collapsed(self) -> None:
        text = _text("chunk:support")
        graph = _graph(build_evidence_id(EvidenceType.GRAPH_RELATIONSHIP, "g"), support_text_ids=[text.evidence_id])

        fused, _diagnostics = EvidenceFusionService(rrf_k=60).fuse(
            vector_evidence=[text],
            graph_evidence=[graph],
            graph_support_evidence=[text],
            top_k=10,
        )

        assert {item.evidence.evidence_type for item in fused} == {
            EvidenceType.TEXT,
            EvidenceType.GRAPH_RELATIONSHIP,
        }
        graph_item = next(item for item in fused if item.evidence.evidence_type == EvidenceType.GRAPH_RELATIONSHIP)
        assert graph_item.cross_store_supported is True

    def test_graph_only_support_text_has_no_branch_rank(self) -> None:
        support = _text("chunk:support")
        graph = _graph(build_evidence_id(EvidenceType.GRAPH_RELATIONSHIP, "g"), support_text_ids=[support.evidence_id])

        fused, _diagnostics = EvidenceFusionService(rrf_k=60).fuse(
            vector_evidence=[],
            graph_evidence=[graph],
            graph_support_evidence=[support],
            top_k=10,
        )

        support_item = next(item for item in fused if item.evidence.evidence_type == EvidenceType.TEXT)
        assert support_item.branch_ranks == {}
        assert support_item.branches == ["graph_support"]


class TestHybridRetrievalService:
    def test_vector_strategy_only_calls_vector_branch(self) -> None:
        vector = _VectorService([_hit("chunk:a")])
        graph = _GraphService([_graph(build_evidence_id(EvidenceType.GRAPH_RELATIONSHIP, "g"))])

        result = _service(vector, graph, _Bridge()).retrieve(
            query="graph rag",
            strategy=RetrievalStrategy.VECTOR,
        )

        assert len(vector.calls) == 1
        assert graph.calls == []
        assert result.vector_result.evidence[0].score_kind == EvidenceScoreKind.VECTOR_SIMILARITY

    def test_graph_strategy_only_calls_graph_branch_and_bridge(self) -> None:
        graph_evidence = _graph(build_evidence_id(EvidenceType.GRAPH_RELATIONSHIP, "g"))
        graph = _GraphService([graph_evidence])
        bridge = _Bridge({graph_evidence.evidence_id: [_text("chunk:support")]})

        result = _service(_VectorService([_hit("chunk:a")]), graph, bridge).retrieve(
            query="explicit graph",
            strategy=RetrievalStrategy.GRAPH,
            graph_operation=GraphSearchOperation.PAPER_METHODS,
            entity_id="paper:arxiv:a",
        )

        assert result.vector_result is None
        assert len(graph.calls) == 1
        assert bridge.calls == [graph_evidence.evidence_id]
        assert result.graph_result.evidence[0].score_kind == EvidenceScoreKind.GRAPH_PATH_CONFIDENCE

    def test_hybrid_calls_both_branches_once_and_preserves_scores(self) -> None:
        text = _hit("chunk:support", score=0.77)
        text_id = build_text_evidence_id("chunk:support", "vector-current")
        graph_evidence = _graph(
            build_evidence_id(EvidenceType.GRAPH_RELATIONSHIP, "g"),
            support_text_ids=[text_id],
        )

        result = _service(
            _VectorService([text]),
            _GraphService([graph_evidence]),
            _Bridge({graph_evidence.evidence_id: [_text("chunk:support")]}),
        ).retrieve(
            query="hybrid query",
            strategy=RetrievalStrategy.HYBRID,
            graph_operation=GraphSearchOperation.PAPER_METHODS,
            entity_id="paper:arxiv:a",
        )

        assert result.vector_result.evidence[0].score == 0.77
        assert result.vector_result.evidence[0].score_kind == EvidenceScoreKind.VECTOR_SIMILARITY
        assert result.graph_result.evidence[0].score_kind == EvidenceScoreKind.GRAPH_PATH_CONFIDENCE
        assert result.diagnostics["cross_store_links"] == 1

    def test_empty_branches_return_valid_results(self) -> None:
        result = _service(_VectorService([]), _GraphService([]), _Bridge()).retrieve(
            query="empty",
            strategy=RetrievalStrategy.HYBRID,
            graph_operation=GraphSearchOperation.PAPER_METHODS,
            entity_id="paper:arxiv:a",
        )

        assert result.evidence == []
        assert result.evidence_pool.items == []
        assert result.diagnostics["vector_branch_empty"] is True
        assert result.diagnostics["graph_branch_empty"] is True

    def test_empty_vector_branch_still_returns_graph(self) -> None:
        graph_evidence = _graph(build_evidence_id(EvidenceType.GRAPH_RELATIONSHIP, "g"))

        result = _service(_VectorService([]), _GraphService([graph_evidence]), _Bridge()).retrieve(
            query="graph survives",
            strategy=RetrievalStrategy.HYBRID,
            graph_operation=GraphSearchOperation.PAPER_METHODS,
            entity_id="paper:arxiv:a",
        )

        assert [item.evidence.evidence_type for item in result.evidence] == [EvidenceType.GRAPH_RELATIONSHIP]

    def test_empty_graph_branch_still_returns_vector(self) -> None:
        result = _service(_VectorService([_hit("chunk:a")]), _GraphService([]), _Bridge()).retrieve(
            query="vector survives",
            strategy=RetrievalStrategy.HYBRID,
            graph_operation=GraphSearchOperation.PAPER_METHODS,
            entity_id="paper:arxiv:a",
        )

        assert [item.evidence.evidence_type for item in result.evidence] == [EvidenceType.TEXT]

    def test_branch_failure_is_not_silently_hidden(self) -> None:
        with pytest.raises(VectorSearchError):
            _service(_VectorService(error=VectorSearchError("boom")), _GraphService([]), _Bridge()).retrieve(
                query="fail",
                strategy=RetrievalStrategy.HYBRID,
                graph_operation=GraphSearchOperation.PAPER_METHODS,
                entity_id="paper:arxiv:a",
            )

        with pytest.raises(RuntimeError):
            _service(_VectorService([]), _GraphService(error=RuntimeError("graph boom")), _Bridge()).retrieve(
                query="fail graph",
                strategy=RetrievalStrategy.HYBRID,
                graph_operation=GraphSearchOperation.PAPER_METHODS,
                entity_id="paper:arxiv:a",
            )

    def test_top_k_validation_and_pool_order(self) -> None:
        with pytest.raises(HybridRetrievalError):
            _service(_VectorService([]), _GraphService([]), _Bridge()).retrieve(
                query="bad",
                strategy=RetrievalStrategy.VECTOR,
                top_k=0,
            )

        result = _service(
            _VectorService([_hit("chunk:a"), _hit("chunk:b")]),
            _GraphService([]),
            _Bridge(),
        ).retrieve(query="pool", strategy=RetrievalStrategy.VECTOR, top_k=1)

        assert len(result.evidence) == 1
        assert [(item.pool_id, item.evidence.evidence_id) for item in result.evidence_pool.items] == [
            ("E1", result.evidence[0].evidence.evidence_id)
        ]

        max_result = _service(
            _VectorService([_hit(f"chunk:{index}") for index in range(3)]),
            _GraphService([]),
            _Bridge(),
        ).retrieve(query="max", strategy=RetrievalStrategy.VECTOR, top_k=50)
        assert len(max_result.evidence) == 3

        with pytest.raises(HybridRetrievalError):
            _service(_VectorService([]), _GraphService([]), _Bridge()).retrieve(
                query="too much",
                strategy=RetrievalStrategy.VECTOR,
                top_k=51,
            )

    def test_graph_requires_explicit_operation(self) -> None:
        with pytest.raises(HybridRetrievalError):
            _service(_VectorService([]), _GraphService([]), _Bridge()).retrieve(
                query="missing graph operation",
                strategy=RetrievalStrategy.GRAPH,
                entity_type=EntityType.PAPER,
                canonical_name="Paper A",
            )

    def test_graph_only_rejects_vector_filters(self) -> None:
        with pytest.raises(HybridRetrievalError):
            _service(_VectorService([]), _GraphService([]), _Bridge()).retrieve(
                query="graph filter",
                strategy=RetrievalStrategy.GRAPH,
                graph_operation=GraphSearchOperation.PAPER_METHODS,
                entity_id="paper:arxiv:a",
                paper_id="paper:arxiv:a",
            )
