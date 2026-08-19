"""Explicit-strategy hybrid retrieval and deterministic evidence fusion.

Prompt 12 deliberately keeps strategy and graph operation selection outside
this service. Callers choose `vector`, `graph`, or `hybrid`; this module
executes the requested branches once and fuses already-normalized evidence.
"""

import logging
import time

from pydantic import BaseModel, Field

from app.core.exceptions import HybridRetrievalError
from app.domain.enums import EntityType, EvidenceType, RetrievalStrategy
from app.domain.evidence import EvidenceItem, EvidencePool, build_evidence_pool
from app.retrieval.evidence import EvidenceProvenanceBridge, VectorEvidenceAdapter
from app.retrieval.graph_search import GraphRetrievalService, GraphSearchOperation
from app.retrieval.vector_search import VectorSearchService

logger = logging.getLogger(__name__)


class RetrievalBranchResult(BaseModel):
    """Evidence produced by one retrieval branch before fusion."""

    strategy: RetrievalStrategy
    evidence: list[EvidenceItem] = Field(default_factory=list)
    duration_ms: int = 0
    warnings: list[str] = Field(default_factory=list)


class FusedEvidenceItem(BaseModel):
    """One ranked evidence item with fusion metadata."""

    evidence: EvidenceItem
    fusion_score: float
    branch_ranks: dict[str, int] = Field(default_factory=dict)
    branches: list[str] = Field(default_factory=list)
    cross_store_supported: bool = False


class HybridRetrievalResult(BaseModel):
    """Final deterministic retrieval result returned by the API."""

    query: str
    strategy: RetrievalStrategy
    evidence: list[FusedEvidenceItem] = Field(default_factory=list)
    evidence_pool: EvidencePool
    vector_result: RetrievalBranchResult | None = None
    graph_result: RetrievalBranchResult | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)


class EvidenceFusionService:
    """Fuse unified evidence using reciprocal rank fusion (RRF)."""

    def __init__(self, *, rrf_k: int) -> None:
        if rrf_k <= 0:
            raise HybridRetrievalError("HYBRID_RRF_K must be > 0")
        self._rrf_k = rrf_k

    def fuse(
        self,
        *,
        vector_evidence: list[EvidenceItem],
        graph_evidence: list[EvidenceItem],
        graph_support_evidence: list[EvidenceItem],
        top_k: int,
    ) -> tuple[list[FusedEvidenceItem], dict]:
        registry: dict[str, EvidenceItem] = {}
        branch_ranks: dict[str, dict[str, int]] = {}
        branches: dict[str, set[str]] = {}
        scores: dict[str, float] = {}

        for branch_name, evidence_items in [
            (RetrievalStrategy.VECTOR.value, vector_evidence),
            (RetrievalStrategy.GRAPH.value, graph_evidence),
        ]:
            for rank, evidence in enumerate(evidence_items, start=1):
                self._register(registry, evidence)
                branch_ranks.setdefault(evidence.evidence_id, {})[branch_name] = rank
                branches.setdefault(evidence.evidence_id, set()).add(branch_name)
                scores[evidence.evidence_id] = scores.get(evidence.evidence_id, 0.0) + (
                    1.0 / (self._rrf_k + rank)
                )

        for evidence in graph_support_evidence:
            self._register(registry, evidence)
            branches.setdefault(evidence.evidence_id, set()).add("graph_support")
            scores.setdefault(evidence.evidence_id, 0.0)

        vector_ids = {item.evidence_id for item in vector_evidence}
        cross_store_links = 0
        fused: list[FusedEvidenceItem] = []
        for evidence_id, evidence in registry.items():
            cross_store_supported = (
                evidence.evidence_type in {EvidenceType.GRAPH_RELATIONSHIP, EvidenceType.GRAPH_PATH}
                and bool(set(evidence.supporting_text_evidence_ids) & vector_ids)
            )
            if cross_store_supported:
                cross_store_links += 1
                evidence = evidence.model_copy(
                    update={
                        "metadata": {
                            **evidence.metadata,
                            "cross_store_supported": True,
                        }
                    }
                )
            fused.append(
                FusedEvidenceItem(
                    evidence=evidence,
                    fusion_score=scores[evidence_id],
                    branch_ranks=branch_ranks.get(evidence_id, {}),
                    branches=sorted(branches.get(evidence_id, set())),
                    cross_store_supported=cross_store_supported,
                )
            )

        fused.sort(
            key=lambda item: (
                -item.fusion_score,
                _best_branch_rank(item.branch_ranks),
                item.evidence.evidence_id,
            )
        )
        duplicate_count = len(vector_evidence) + len(graph_evidence) + len(graph_support_evidence) - len(
            registry
        )
        diagnostics = {
            "fusion_method": "rrf",
            "rrf_k": self._rrf_k,
            "unique_evidence": len(registry),
            "duplicate_evidence_removed": duplicate_count,
            "cross_store_links": cross_store_links,
        }
        return fused[:top_k], diagnostics

    def _register(self, registry: dict[str, EvidenceItem], evidence: EvidenceItem) -> None:
        registry.setdefault(evidence.evidence_id, evidence)


class HybridRetrievalService:
    """Execute explicit retrieval strategies and fuse normalized evidence."""

    def __init__(
        self,
        *,
        vector_service: VectorSearchService,
        graph_service: GraphRetrievalService,
        provenance_bridge: EvidenceProvenanceBridge,
        fusion_service: EvidenceFusionService,
        default_top_k: int,
        max_top_k: int,
    ) -> None:
        self._vector_service = vector_service
        self._graph_service = graph_service
        self._bridge = provenance_bridge
        self._fusion = fusion_service
        self._default_top_k = default_top_k
        self._max_top_k = max_top_k
        self._vector_adapter = VectorEvidenceAdapter()

    def retrieve(
        self,
        *,
        query: str,
        strategy: RetrievalStrategy,
        top_k: int | None = None,
        vector_top_k: int | None = None,
        graph_operation: GraphSearchOperation | None = None,
        entity_id: str | None = None,
        entity_type: EntityType | None = None,
        canonical_name: str | None = None,
        graph_depth: int | None = None,
        graph_limit: int | None = None,
        paper_id: str | None = None,
        paper_version_id: str | None = None,
        section_type: str | None = None,
    ) -> HybridRetrievalResult:
        started = time.monotonic()
        final_top_k = self._validate_top_k(top_k)
        if not query.strip():
            raise HybridRetrievalError("query must not be blank")
        self._validate_strategy_request(
            strategy=strategy,
            graph_operation=graph_operation,
            paper_id=paper_id,
            paper_version_id=paper_version_id,
            section_type=section_type,
        )

        vector_result = None
        graph_result = None
        graph_support_evidence: list[EvidenceItem] = []
        warnings: list[str] = []

        if strategy in {RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID}:
            vector_result = self._run_vector_branch(
                query=query,
                top_k=vector_top_k,
                paper_id=paper_id,
                paper_version_id=paper_version_id,
                section_type=section_type,
            )
            warnings.extend(vector_result.warnings)

        if strategy in {RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID}:
            graph_result, graph_support_evidence = self._run_graph_branch(
                graph_operation=graph_operation,
                entity_id=entity_id,
                entity_type=entity_type,
                canonical_name=canonical_name,
                graph_depth=graph_depth,
                graph_limit=graph_limit,
            )
            warnings.extend(graph_result.warnings)

        vector_evidence = vector_result.evidence if vector_result else []
        graph_evidence = graph_result.evidence if graph_result else []
        fused, fusion_diagnostics = self._fusion.fuse(
            vector_evidence=vector_evidence,
            graph_evidence=graph_evidence,
            graph_support_evidence=graph_support_evidence,
            top_k=final_top_k,
        )
        evidence_pool = build_evidence_pool([item.evidence for item in fused])
        duration_ms = int((time.monotonic() - started) * 1000)
        diagnostics = {
            **fusion_diagnostics,
            "strategy": strategy.value,
            "vector_candidates": len(vector_evidence),
            "graph_candidates": len(graph_evidence),
            "graph_support_evidence": len(graph_support_evidence),
            "final_evidence": len(fused),
            "duration_ms": duration_ms,
        }
        if vector_result is not None and not vector_evidence:
            diagnostics["vector_branch_empty"] = True
        if graph_result is not None and not graph_evidence:
            diagnostics["graph_branch_empty"] = True

        logger.info(
            "hybrid retrieval strategy=%s vector_candidate_count=%d graph_candidate_count=%d "
            "final_count=%d cross_store_links=%d fusion_method=%s duration_ms=%d status=ok",
            strategy.value,
            len(vector_evidence),
            len(graph_evidence),
            len(fused),
            diagnostics["cross_store_links"],
            diagnostics["fusion_method"],
            duration_ms,
        )
        return HybridRetrievalResult(
            query=query,
            strategy=strategy,
            evidence=fused,
            evidence_pool=evidence_pool,
            vector_result=vector_result,
            graph_result=graph_result,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def _validate_top_k(self, top_k: int | None) -> int:
        resolved = self._default_top_k if top_k is None else top_k
        if resolved <= 0 or resolved > self._max_top_k:
            raise HybridRetrievalError(
                f"top_k must be between 1 and {self._max_top_k} (got {resolved})"
            )
        return resolved

    def _validate_strategy_request(
        self,
        *,
        strategy: RetrievalStrategy,
        graph_operation: GraphSearchOperation | None,
        paper_id: str | None,
        paper_version_id: str | None,
        section_type: str | None,
    ) -> None:
        if strategy not in {RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID}:
            raise HybridRetrievalError(f"unsupported retrieval strategy {strategy.value}")
        if strategy in {RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID} and graph_operation is None:
            raise HybridRetrievalError(f"{strategy.value} retrieval requires graph_operation")
        if strategy == RetrievalStrategy.GRAPH and any([paper_id, paper_version_id, section_type]):
            raise HybridRetrievalError(
                "paper_id, paper_version_id, and section_type filters apply to vector retrieval only"
            )

    def _run_vector_branch(
        self,
        *,
        query: str,
        top_k: int | None,
        paper_id: str | None,
        paper_version_id: str | None,
        section_type: str | None,
    ) -> RetrievalBranchResult:
        started = time.monotonic()
        hits = self._vector_service.search(
            query,
            top_k=top_k,
            paper_id=paper_id,
            paper_version_id=paper_version_id,
            section_type=section_type,
        )
        evidence = [self._vector_adapter.from_hit(hit) for hit in hits]
        return RetrievalBranchResult(
            strategy=RetrievalStrategy.VECTOR,
            evidence=evidence,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    def _run_graph_branch(
        self,
        *,
        graph_operation: GraphSearchOperation,
        entity_id: str | None,
        entity_type: EntityType | None,
        canonical_name: str | None,
        graph_depth: int | None,
        graph_limit: int | None,
    ) -> tuple[RetrievalBranchResult, list[EvidenceItem]]:
        started = time.monotonic()
        graph = self._graph_service.search(
            operation=graph_operation,
            entity_id=entity_id,
            entity_type=entity_type,
            canonical_name=canonical_name,
            depth=graph_depth,
            limit=graph_limit,
        )
        bridged_graph: list[EvidenceItem] = []
        support_text: list[EvidenceItem] = []
        warnings: list[str] = []
        for evidence in graph.evidence:
            bridge_result = self._bridge.resolve_graph_evidence_sources(evidence)
            bridged_graph.append(bridge_result.graph_evidence)
            support_text.extend(bridge_result.text_evidence)
            warnings.extend(bridge_result.warnings)
        return (
            RetrievalBranchResult(
                strategy=RetrievalStrategy.GRAPH,
                evidence=bridged_graph,
                duration_ms=int((time.monotonic() - started) * 1000),
                warnings=warnings,
            ),
            support_text,
        )


def _best_branch_rank(branch_ranks: dict[str, int]) -> int:
    if not branch_ranks:
        return 1_000_000
    return min(branch_ranks.values())
