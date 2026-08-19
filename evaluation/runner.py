"""Read-only runner for deterministic retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from app.core.config import Settings
from app.core.exceptions import GraphEntityNotFoundError
from app.domain.enums import RetrievalStrategy
from app.domain.evidence import EvidenceItem
from app.retrieval.hybrid import FusedEvidenceItem, HybridRetrievalService

from evaluation.metrics import (
    aggregate_strategy_results,
    compute_ranking_metrics,
    compute_structural_metrics,
)
from evaluation.models import (
    EvaluationFailureCategory,
    RetrievedEvidenceRecord,
    RetrievalBenchmarkCase,
    RetrievalBenchmarkDataset,
    RetrievalEvaluationCaseResult,
    RetrievalEvaluationCategory,
    RetrievalEvaluationReport,
    StrategyCaseResult,
)


DEFAULT_K_VALUES = [1, 3, 5, 10]


def load_benchmark(path: str | Path) -> RetrievalBenchmarkDataset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RetrievalBenchmarkDataset.model_validate(data)


def benchmark_checksum(dataset: RetrievalBenchmarkDataset) -> str:
    canonical = json.dumps(dataset.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_fingerprint(*, checksum: str, settings: Settings, k_values: list[int]) -> str:
    canonical = {
        "benchmark_checksum": checksum,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_normalize": settings.EMBEDDING_NORMALIZE,
        "canonicalization_version": settings.CANONICALIZATION_VERSION,
        "extraction_version": settings.EXTRACTION_VERSION,
        "rrf_k": settings.HYBRID_RRF_K,
        "k_values": k_values,
        "vector_top_k": settings.VECTOR_SEARCH_DEFAULT_TOP_K,
        "graph_limit": settings.GRAPH_DEFAULT_LIMIT,
        "hybrid_top_k": settings.HYBRID_DEFAULT_TOP_K,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class RetrievalEvaluationRunner:
    """Run benchmark cases through the existing explicit retrieval service."""

    def __init__(
        self,
        retrieval_service: HybridRetrievalService,
        *,
        settings: Settings,
        k_values: list[int] | None = None,
        vector_candidate_count: int | None = None,
        graph_candidate_count: int | None = None,
        hybrid_top_k: int | None = None,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._settings = settings
        self._k_values = k_values or DEFAULT_K_VALUES
        self._vector_candidate_count = vector_candidate_count or settings.VECTOR_SEARCH_DEFAULT_TOP_K
        self._graph_candidate_count = graph_candidate_count or settings.GRAPH_DEFAULT_LIMIT
        self._hybrid_top_k = hybrid_top_k or settings.HYBRID_DEFAULT_TOP_K
        self._metadata_overrides = metadata_overrides or {}

    def run_case(self, case: RetrievalBenchmarkCase) -> RetrievalEvaluationCaseResult:
        strategy_results = {
            strategy.value: self._run_strategy(case, strategy)
            for strategy in [RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID]
        }
        return RetrievalEvaluationCaseResult(
            case_id=case.id,
            category=case.category,
            strategy_results=strategy_results,
            comparison=self._comparison(strategy_results),
        )

    def run_dataset(self, dataset: RetrievalBenchmarkDataset) -> RetrievalEvaluationReport:
        checksum = benchmark_checksum(dataset)
        case_results = [self.run_case(case) for case in dataset.cases]
        overall = self._aggregate(case_results)
        by_category: dict[str, dict[str, Any]] = {}
        for category in RetrievalEvaluationCategory:
            category_results = [result for result in case_results if result.category == category]
            by_category[category.value] = self._aggregate(category_results)

        return RetrievalEvaluationReport(
            benchmark_version=dataset.benchmark_version,
            benchmark_checksum=checksum,
            run_fingerprint=run_fingerprint(checksum=checksum, settings=self._settings, k_values=self._k_values),
            k_values=self._k_values,
            generated_at=datetime.now(UTC).isoformat(),
            case_results=case_results,
            overall=overall,
            by_category=by_category,
            hybrid_wins=[result.case_id for result in case_results if _strategy_hit(result, "hybrid") and not _strategy_hit(result, "vector")],
            vector_wins=[
                result.case_id
                for result in case_results
                if _strategy_hit(result, "vector")
                and (
                    not result.strategy_results["hybrid"].applicable
                    or result.strategy_results["vector"].metrics.mrr >= result.strategy_results["hybrid"].metrics.mrr
                )
            ],
            graph_wins=[result.case_id for result in case_results if _strategy_hit(result, "graph") and not _strategy_hit(result, "vector")],
            all_failures=[result.case_id for result in case_results if result.comparison == "ALL_FAIL"],
            metadata={
                "embedding_model": self._settings.EMBEDDING_MODEL,
                "embedding_provider": self._settings.EMBEDDING_PROVIDER,
                "embedding_config_fingerprint": "configured-at-index-time",
                "canonicalization_version": self._settings.CANONICALIZATION_VERSION,
                "extraction_version": self._settings.EXTRACTION_VERSION,
                "rrf_k": self._settings.HYBRID_RRF_K,
                "vector_candidate_count": self._vector_candidate_count,
                "graph_candidate_count": self._graph_candidate_count,
                "hybrid_final_top_k": self._hybrid_top_k,
                "llm_calls": 0,
                "llm_tokens": 0,
                **self._metadata_overrides,
            },
        )

    def _run_strategy(self, case: RetrievalBenchmarkCase, strategy: RetrievalStrategy) -> StrategyCaseResult:
        if strategy in {RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID} and case.graph_request is None:
            return StrategyCaseResult(
                strategy=strategy,
                applicable=False,
                failure_reason=EvaluationFailureCategory.GRAPH_NOT_APPLICABLE,
            )

        graph_request = case.graph_request
        started = time.monotonic()
        try:
            result = self._retrieval_service.retrieve(
                query=case.question,
                strategy=strategy,
                vector_top_k=self._vector_candidate_count,
                top_k=self._hybrid_top_k,
                graph_operation=graph_request.operation if graph_request else None,
                entity_id=graph_request.entity_id if graph_request else None,
                entity_type=graph_request.entity_type if graph_request else None,
                canonical_name=graph_request.canonical_name if graph_request else None,
                graph_depth=graph_request.depth if graph_request else None,
                graph_limit=graph_request.limit if graph_request and graph_request.limit else self._graph_candidate_count,
            )
            latency_ms = (time.monotonic() - started) * 1000
            fused_items = result.evidence
            evidence = [item.evidence for item in fused_items]
            metrics, relevant_count, matched_by_rank = compute_ranking_metrics(
                evidence, case.expected_targets, k_values=self._k_values
            )
            structural_metrics = compute_structural_metrics(evidence, case.expected_targets)
            return StrategyCaseResult(
                strategy=strategy,
                applicable=True,
                retrieved_count=len(evidence),
                relevant_count=relevant_count,
                metrics=metrics,
                structural_metrics=structural_metrics,
                latency_ms=latency_ms,
                failure_reason=_failure_reason(evidence, relevant_count, structural_metrics),
                evidence=[
                    _record_fused_evidence(item, rank, matches)
                    for rank, (item, matches) in enumerate(zip(fused_items, matched_by_rank, strict=True), start=1)
                ],
                diagnostics=result.diagnostics,
            )
        except GraphEntityNotFoundError as exc:
            return _failed_result(strategy, started, EvaluationFailureCategory.ENTITY_NOT_FOUND, str(exc))
        except Exception as exc:
            return _failed_result(strategy, started, EvaluationFailureCategory.RETRIEVAL_ERROR, str(exc))

    def _aggregate(self, case_results: list[RetrievalEvaluationCaseResult]) -> dict[str, Any]:
        return {
            strategy.value: aggregate_strategy_results(
                [case.strategy_results[strategy.value] for case in case_results],
                k_values=self._k_values,
            )
            for strategy in [RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID]
        }

    def _comparison(self, strategy_results: dict[str, StrategyCaseResult]) -> str:
        vector = strategy_results[RetrievalStrategy.VECTOR.value]
        graph = strategy_results[RetrievalStrategy.GRAPH.value]
        hybrid = strategy_results[RetrievalStrategy.HYBRID.value]
        vector_hit = vector.applicable and vector.metrics.hit_at_k.get(5, 0.0) > 0
        graph_hit = graph.applicable and graph.metrics.hit_at_k.get(5, 0.0) > 0
        hybrid_hit = hybrid.applicable and hybrid.metrics.hit_at_k.get(5, 0.0) > 0
        if hybrid_hit and not vector_hit:
            return "HYBRID_WINS"
        if graph_hit and not vector_hit:
            return "GRAPH_WINS"
        if vector_hit and (
            not hybrid.applicable
            or vector.metrics.mrr >= hybrid.metrics.mrr
            or hybrid.metrics.hit_at_k.get(5, 0.0) == 0
        ):
            return "VECTOR_WINS"
        if not vector_hit and not graph_hit and not hybrid_hit:
            return "ALL_FAIL"
        return "NO_CLEAR_WIN"


def validate_benchmark_file(path: str | Path) -> RetrievalBenchmarkDataset:
    adapter = TypeAdapter(RetrievalBenchmarkDataset)
    return adapter.validate_python(json.loads(Path(path).read_text(encoding="utf-8")))


def _record_fused_evidence(
    item: FusedEvidenceItem,
    rank: int,
    matched_target_ids: list[str],
) -> RetrievedEvidenceRecord:
    return RetrievedEvidenceRecord(
        evidence_id=item.evidence.evidence_id,
        evidence_type=item.evidence.evidence_type.value,
        matched_target_ids=matched_target_ids,
        rank=rank,
        branch_ranks=item.branch_ranks,
        fusion_score=item.fusion_score,
        cross_store_supported=item.cross_store_supported,
    )


def _failure_reason(
    evidence: list[EvidenceItem],
    relevant_count: int,
    structural_metrics,
) -> EvaluationFailureCategory | None:
    if not evidence:
        return EvaluationFailureCategory.NO_RESULTS
    if structural_metrics.provenance_completeness < 1.0:
        return EvaluationFailureCategory.PROVENANCE_INCOMPLETE
    if relevant_count == 0:
        return EvaluationFailureCategory.RELEVANT_NOT_IN_TOP_K
    return None


def _failed_result(
    strategy: RetrievalStrategy,
    started: float,
    failure: EvaluationFailureCategory,
    message: str,
) -> StrategyCaseResult:
    return StrategyCaseResult(
        strategy=strategy,
        applicable=True,
        latency_ms=(time.monotonic() - started) * 1000,
        failure_reason=failure,
        diagnostics={"error": message},
    )


def _strategy_hit(result: RetrievalEvaluationCaseResult, strategy: str) -> bool:
    strategy_result = result.strategy_results[strategy]
    return strategy_result.applicable and strategy_result.metrics.hit_at_k.get(5, 0.0) > 0.0
