"""End-to-end evaluation runner for RAG architecture variants."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from app.core.config import Settings
from app.domain.enums import ConfidenceLevel, RetrievalStrategy
from app.domain.evidence import EvidenceItem, build_evidence_pool
from app.generation.answer import AnswerContextBuilder, GroundedAnswerGenerator, answer_generation_config_fingerprint
from app.generation.citations import CitationValidator
from app.generation.grounding import GroundingDecisionService
from app.retrieval.critic import EvidenceAssessment, EvidenceCoverage, RefinementType
from app.retrieval.hybrid import HybridRetrievalService
from app.retrieval.workflow import RetrievalWorkflowService
from evaluation.end_to_end_models import (
    EndToEndBenchmark,
    EndToEndBenchmarkCase,
    EndToEndCaseResult,
    EndToEndCategory,
    EndToEndEvaluationReport,
    EndToEndSystem,
    EndToEndSystemMetrics,
    PairwiseComparison,
)
from evaluation.metrics import compute_ranking_metrics, percentile


END_TO_END_EVALUATION_VERSION = "v1"


def load_end_to_end_benchmark(path: str | Path) -> EndToEndBenchmark:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EndToEndBenchmark.model_validate(data)


def validate_end_to_end_benchmark_file(path: str | Path) -> EndToEndBenchmark:
    adapter = TypeAdapter(EndToEndBenchmark)
    return adapter.validate_python(json.loads(Path(path).read_text(encoding="utf-8")))


def end_to_end_benchmark_checksum(dataset: EndToEndBenchmark) -> str:
    canonical = json.dumps(dataset.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def end_to_end_run_fingerprint(*, checksum: str, settings: Settings) -> str:
    canonical = {
        "benchmark_checksum": checksum,
        "evaluation_version": END_TO_END_EVALUATION_VERSION,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_normalize": settings.EMBEDDING_NORMALIZE,
        "canonicalization_version": settings.CANONICALIZATION_VERSION,
        "extraction_version": settings.EXTRACTION_VERSION,
        "rrf_k": settings.HYBRID_RRF_K,
        "query_planner_version": settings.QUERY_PLANNER_VERSION,
        "query_planner_rules_version": settings.QUERY_PLANNER_RULES_VERSION,
        "critic_rules_version": settings.EVIDENCE_CRITIC_RULES_VERSION,
        "answer_prompt_version": settings.ANSWER_GENERATION_PROMPT_VERSION,
        "answer_schema_version": settings.ANSWER_GENERATION_SCHEMA_VERSION,
        "citation_validator_version": settings.CITATION_VALIDATOR_VERSION,
        "grounding_rules_version": settings.GROUNDING_RULES_VERSION,
        "confidence_rules_version": settings.CONFIDENCE_RULES_VERSION,
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


class EndToEndEvaluationRunner:
    """Compare static baselines and the production agentic workflow."""

    def __init__(
        self,
        *,
        retrieval_service: HybridRetrievalService,
        answer_context_builder: AnswerContextBuilder,
        answer_generator: GroundedAnswerGenerator,
        citation_validator: CitationValidator,
        grounding_service: GroundingDecisionService,
        agentic_workflow: RetrievalWorkflowService,
        settings: Settings,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._answer_context_builder = answer_context_builder
        self._answer_generator = answer_generator
        self._citation_validator = citation_validator
        self._grounding_service = grounding_service
        self._agentic_workflow = agentic_workflow
        self._settings = settings
        self._metadata_overrides = metadata_overrides or {}

    def run_dataset(self, dataset: EndToEndBenchmark) -> EndToEndEvaluationReport:
        checksum = end_to_end_benchmark_checksum(dataset)
        results = [
            result
            for case in dataset.cases
            for result in self.run_case(case)
        ]
        overall = {
            system.value: _aggregate([result for result in results if result.system == system])
            for system in EndToEndSystem
        }
        by_category: dict[str, dict[str, EndToEndSystemMetrics]] = {}
        for category in EndToEndCategory:
            category_results = [result for result in results if result.category == category]
            by_category[category.value] = {
                system.value: _aggregate([result for result in category_results if result.system == system])
                for system in EndToEndSystem
            }
        pairwise = {
            "agentic_vs_vector": _pairwise(results, EndToEndSystem.AGENTIC_HYBRID_RAG, EndToEndSystem.VECTOR_RAG),
            "hybrid_vs_vector": _pairwise(results, EndToEndSystem.HYBRID_RAG, EndToEndSystem.VECTOR_RAG),
            "graph_vs_vector": _pairwise(results, EndToEndSystem.GRAPH_RAG, EndToEndSystem.VECTOR_RAG),
            "agentic_vs_hybrid": _pairwise(results, EndToEndSystem.AGENTIC_HYBRID_RAG, EndToEndSystem.HYBRID_RAG),
        }
        return EndToEndEvaluationReport(
            benchmark_version=dataset.benchmark_version,
            benchmark_checksum=checksum,
            run_fingerprint=end_to_end_run_fingerprint(checksum=checksum, settings=self._settings),
            generated_at=datetime.now(UTC).isoformat(),
            case_results=results,
            overall=overall,
            by_category=by_category,
            pairwise=pairwise,
            agentic_wins=_agentic_wins(results),
            agentic_losses=_agentic_losses(results),
            failure_analysis=_failure_analysis(results),
            metadata={
                "evaluation_version": END_TO_END_EVALUATION_VERSION,
                "embedding_provider": self._settings.EMBEDDING_PROVIDER,
                "embedding_model": self._settings.EMBEDDING_MODEL,
                "embedding_normalize": self._settings.EMBEDDING_NORMALIZE,
                "canonicalization_version": self._settings.CANONICALIZATION_VERSION,
                "extraction_version": self._settings.EXTRACTION_VERSION,
                "rrf_k": self._settings.HYBRID_RRF_K,
                "planner_version": self._settings.QUERY_PLANNER_VERSION,
                "critic_rules_version": self._settings.EVIDENCE_CRITIC_RULES_VERSION,
                "answer_generation_prompt_version": self._settings.ANSWER_GENERATION_PROMPT_VERSION,
                "citation_validator_version": self._settings.CITATION_VALIDATOR_VERSION,
                "grounding_rules_version": self._settings.GROUNDING_RULES_VERSION,
                "confidence_rules_version": self._settings.CONFIDENCE_RULES_VERSION,
                "provider": self._answer_generator.provider_name,
                "model": self._answer_generator.model_name,
                "fake_provider_note": "controlled fake providers validate software contracts, not live model quality",
                **self._metadata_overrides,
            },
        )

    def run_case(self, case: EndToEndBenchmarkCase) -> list[EndToEndCaseResult]:
        return [
            self._run_static_baseline(case, EndToEndSystem.VECTOR_RAG, RetrievalStrategy.VECTOR),
            self._run_static_baseline(case, EndToEndSystem.GRAPH_RAG, RetrievalStrategy.GRAPH),
            self._run_static_baseline(case, EndToEndSystem.HYBRID_RAG, RetrievalStrategy.HYBRID),
            self._run_agentic(case),
        ]

    def _run_static_baseline(
        self,
        case: EndToEndBenchmarkCase,
        system: EndToEndSystem,
        strategy: RetrievalStrategy,
    ) -> EndToEndCaseResult:
        if strategy in {RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID} and case.graph_request is None:
            return _not_applicable(case, system, "graph_request_missing")

        started = time.monotonic()
        llm_before = getattr(self._answer_generator, "calls", None)
        try:
            graph_request = case.graph_request
            retrieval = self._retrieval_service.retrieve(
                query=case.question,
                strategy=strategy,
                top_k=self._settings.HYBRID_DEFAULT_TOP_K,
                vector_top_k=self._settings.VECTOR_SEARCH_DEFAULT_TOP_K,
                graph_operation=graph_request.operation if graph_request else None,
                entity_id=graph_request.entity_id if graph_request else None,
                entity_type=graph_request.entity_type if graph_request else None,
                canonical_name=graph_request.canonical_name if graph_request else None,
                graph_depth=graph_request.depth if graph_request else None,
                graph_limit=graph_request.limit if graph_request and graph_request.limit else self._settings.GRAPH_DEFAULT_LIMIT,
            )
            evidence = [item.evidence for item in retrieval.evidence]
            assessment = _baseline_assessment(evidence)
            pool = build_evidence_pool(evidence)
            final = None
            citation_validation = None
            if evidence:
                fingerprint = answer_generation_config_fingerprint(
                    settings=self._settings,
                    provider_name=self._answer_generator.provider_name,
                    model_name=self._answer_generator.model_name,
                    temperature=self._answer_generator.temperature,
                )
                context = self._answer_context_builder.build(
                    query=case.question,
                    analysis=None,
                    evidence_pool=pool,
                    generation_config_fingerprint=fingerprint,
                )
                generated = self._answer_generator.generate(context=context)
                validated = self._citation_validator.validate(
                    generated_answer=generated,
                    evidence_pool=pool,
                    answer_context=context,
                )
                citation_validation = validated.citation_validation
                internal_status = (
                    "CITATION_VALIDATION_FAILED"
                    if citation_validation.validation_status.value in {"invalid", "no_citations"}
                    else "SUCCESS"
                )
                final = self._grounding_service.decide(
                    query=case.question,
                    internal_status=internal_status,
                    evidence=evidence,
                    evidence_assessment=assessment,
                    citation_validation=citation_validation,
                    validated_answer=validated,
                    retrieval_round=1,
                    warnings=[*retrieval.warnings, *citation_validation.warnings],
                )
            else:
                final = self._grounding_service.decide(
                    query=case.question,
                    internal_status="EMPTY_EVIDENCE",
                    evidence=[],
                    evidence_assessment=assessment,
                    citation_validation=None,
                    validated_answer=None,
                    retrieval_round=1,
                    warnings=retrieval.warnings,
                )
            latency_ms = (time.monotonic() - started) * 1000
            return _case_result(
                case=case,
                system=system,
                selected_strategy=strategy.value,
                graph_operation=graph_request.operation.value if graph_request else None,
                evidence=evidence,
                answer_status=final.status.value,
                answer=final.answer,
                confidence=final.confidence,
                trusted_citations=len(final.citations),
                citations=final.citations,
                citation_validation=citation_validation,
                retrieval_rounds=1,
                latency_ms=latency_ms,
                llm_calls=_call_delta(llm_before, getattr(self._answer_generator, "calls", None), default=1 if evidence else 0),
                diagnostics={"retrieval": retrieval.diagnostics, "grounding": final.grounding.model_dump(mode="json")},
            )
        except Exception as exc:
            return _error_result(case, system, started, str(exc), selected_strategy=strategy.value)

    def _run_agentic(self, case: EndToEndBenchmarkCase) -> EndToEndCaseResult:
        started = time.monotonic()
        result = self._agentic_workflow.run(case.question)
        latency_ms = (time.monotonic() - started) * 1000
        final = result.final_answer
        answer_status = final.status.value if final else "failed"
        confidence = result.confidence or ConfidenceLevel.INSUFFICIENT_EVIDENCE
        return _case_result(
            case=case,
            system=EndToEndSystem.AGENTIC_HYBRID_RAG,
            selected_strategy=result.retrieval_plan.strategy.value if result.retrieval_plan else None,
            planner_intent=result.analysis.intent.value if result.analysis else None,
            graph_operation=result.retrieval_plan.graph_operation if result.retrieval_plan else None,
            evidence=result.evidence,
            answer_status=answer_status,
            answer=result.answer or "",
            confidence=confidence,
            trusted_citations=len(result.citations),
            citations=result.citations,
            citation_validation=result.citation_validation,
            retrieval_rounds=result.retrieval_round,
            latency_ms=latency_ms,
            llm_calls=_agentic_llm_calls(result),
            token_usage=_token_usage(result.answer_generation_metadata),
            refined=result.retrieval_round > 1,
            refinement_success=result.retrieval_round > 1 and result.evidence_sufficient is True,
            no_new_evidence_refinement=any(event.status == "no_new_evidence" for event in result.trace),
            max_round_insufficient=result.retrieval_round >= self._settings.MAX_RETRIEVAL_ROUNDS and result.evidence_sufficient is False,
            failure_reason=_agentic_failure_reason(result) if result.final_status and result.final_status.value != "answered" else None,
            diagnostics={
                "workflow_status": result.status.value,
                "timings": result.timings,
                "trace": [event.model_dump(mode="json") for event in result.trace],
                "grounding": result.grounding.model_dump(mode="json") if result.grounding else None,
            },
        )


def _baseline_assessment(evidence: list[EvidenceItem]) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=bool(evidence),
        coverage=EvidenceCoverage.COMPLETE if evidence else EvidenceCoverage.INSUFFICIENT,
        missing_information=[] if evidence else ["no evidence was retrieved"],
        recommended_refinement_type=RefinementType.NONE,
        deterministic=True,
    )


def _case_result(
    *,
    case: EndToEndBenchmarkCase,
    system: EndToEndSystem,
    selected_strategy: str | None,
    evidence: list[EvidenceItem],
    answer_status: str,
    answer: str,
    confidence: ConfidenceLevel,
    trusted_citations: int,
    citations: list[Any],
    citation_validation: Any,
    retrieval_rounds: int,
    latency_ms: float,
    llm_calls: int,
    graph_operation: str | None = None,
    planner_intent: str | None = None,
    token_usage: dict[str, int] | None = None,
    refined: bool = False,
    refinement_success: bool = False,
    no_new_evidence_refinement: bool = False,
    max_round_insufficient: bool = False,
    failure_reason: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> EndToEndCaseResult:
    metrics, relevant_count, _ = compute_ranking_metrics(evidence, case.expected_targets, k_values=[5])
    evidence_recall = metrics.recall_at_k.get(5, 0.0)
    answer_text = answer.lower()
    fact_text_ok = all(
        fact.expected_text is None or fact.expected_text.lower() in answer_text
        for fact in case.expected_answer_facts
    )
    citation_targets = _citation_target_ids(citations)
    citation_target_ok = all(target in citation_targets for target in case.required_citation_targets)
    target_ok = relevant_count > 0 or not case.expected_targets
    answerable = not case.expected_abstention and not case.expected_disambiguation
    abstained = answer_status in {"abstained", "requires_disambiguation", "failed"}
    disambiguation_correct = case.expected_disambiguation and answer_status == "requires_disambiguation"
    abstention_correct = case.expected_abstention and abstained
    answer_correct = bool(answerable and answer_status == "answered" and target_ok and fact_text_ok and citation_target_ok)
    citation_status = citation_validation.validation_status.value if citation_validation else None
    markers_found = 0
    if citation_validation is not None:
        markers_found = len(citation_validation.valid_markers) + len(citation_validation.invalid_markers)
    citation_validity_rate = 0.0 if markers_found == 0 else len(citation_validation.valid_markers) / markers_found
    trusted_citation_rate = 0.0 if markers_found == 0 else trusted_citations / markers_found
    provenance_completeness = _citation_provenance_completeness(citations)
    if failure_reason is None:
        failure_reason = _failure_reason(
            answerable=answerable,
            answer_correct=answer_correct,
            abstention_correct=abstention_correct,
            disambiguation_correct=disambiguation_correct,
            citation_status=citation_status,
            evidence_recall=evidence_recall,
            abstained=abstained,
        )
    planning_correct, strategy_correct = _planning_correctness(
        case=case,
        system=system,
        selected_strategy=selected_strategy,
        planner_intent=planner_intent,
        graph_operation=graph_operation,
    )
    return EndToEndCaseResult(
        case_id=case.id,
        category=case.category,
        system=system,
        selected_strategy=selected_strategy,
        planner_intent=planner_intent,
        graph_operation=graph_operation,
        planning_correct=planning_correct,
        strategy_correct=strategy_correct,
        retrieval_rounds=retrieval_rounds,
        evidence_count=len(evidence),
        evidence_recall=evidence_recall,
        answer_status=answer_status,
        answer_correct=answer_correct,
        abstained=abstained,
        abstention_correct=abstention_correct,
        expected_disambiguation=case.expected_disambiguation,
        disambiguation_correct=disambiguation_correct,
        confidence=confidence,
        trusted_citations=trusted_citations,
        citation_status=citation_status,
        citation_validity_rate=citation_validity_rate,
        trusted_citation_rate=trusted_citation_rate,
        provenance_completeness=provenance_completeness,
        latency_ms=latency_ms,
        llm_calls=llm_calls,
        token_usage=token_usage or {},
        refined=refined,
        refinement_success=refinement_success,
        no_new_evidence_refinement=no_new_evidence_refinement,
        max_round_insufficient=max_round_insufficient,
        failure_reason=failure_reason,
        diagnostics=diagnostics or {},
    )


def _not_applicable(case: EndToEndBenchmarkCase, system: EndToEndSystem, reason: str) -> EndToEndCaseResult:
    return EndToEndCaseResult(
        case_id=case.id,
        category=case.category,
        system=system,
        answer_status="not_applicable",
        failure_reason=reason,
        diagnostics={"applicable": False},
    )


def _error_result(
    case: EndToEndBenchmarkCase,
    system: EndToEndSystem,
    started: float,
    message: str,
    *,
    selected_strategy: str | None,
) -> EndToEndCaseResult:
    return EndToEndCaseResult(
        case_id=case.id,
        category=case.category,
        system=system,
        selected_strategy=selected_strategy,
        answer_status="failed",
        abstained=True,
        confidence=ConfidenceLevel.INSUFFICIENT_EVIDENCE,
        latency_ms=(time.monotonic() - started) * 1000,
        failure_reason="execution_error",
        diagnostics={"error": message},
    )


def _aggregate(results: list[EndToEndCaseResult]) -> EndToEndSystemMetrics:
    metrics = EndToEndSystemMetrics(cases=len(results))
    if not results:
        return metrics
    answerable = [r for r in results if not r.expected_disambiguation and r.answer_status != "not_applicable"]
    expected_abstention = [r for r in results if r.category == EndToEndCategory.UNANSWERABLE and r.answer_status != "not_applicable"]
    answered = [r for r in results if r.answer_status == "answered"]
    high_answers = [r for r in answered if r.confidence == ConfidenceLevel.HIGH]
    refined = [r for r in results if r.refined]
    latencies = sorted(r.latency_ms for r in results if r.answer_status != "not_applicable")
    metrics.answer_accuracy = _rate([r.answer_correct for r in answerable if r.category != EndToEndCategory.UNANSWERABLE])
    metrics.correct_abstention_rate = _rate([r.abstention_correct for r in expected_abstention])
    metrics.false_answer_rate = _rate([r.answer_status == "answered" for r in expected_abstention])
    metrics.grounded_answer_rate = _rate([
        r.answer_status == "answered" and r.trusted_citations > 0 and r.confidence != ConfidenceLevel.INSUFFICIENT_EVIDENCE
        for r in answerable
        if r.category != EndToEndCategory.UNANSWERABLE
    ])
    metrics.evidence_recall = statistics.fmean(r.evidence_recall for r in answerable) if answerable else 0.0
    metrics.citation_validity_rate = statistics.fmean(r.citation_validity_rate for r in answered) if answered else 0.0
    metrics.trusted_citation_rate = statistics.fmean(r.trusted_citation_rate for r in answered) if answered else 0.0
    planning_values = [r.planning_correct for r in results if r.planning_correct is not None]
    strategy_values = [r.strategy_correct for r in results if r.strategy_correct is not None]
    metrics.planning_accuracy = _rate(planning_values) if planning_values else None
    metrics.strategy_accuracy = _rate(strategy_values) if strategy_values else None
    metrics.refinement_rate = len(refined) / len(results)
    metrics.refinement_success_rate = _rate([r.refinement_success for r in refined])
    metrics.high_confidence_error_rate = _rate([not r.answer_correct for r in high_answers])
    metrics.provenance_completeness = statistics.fmean(r.provenance_completeness for r in answered) if answered else 0.0
    metrics.mean_latency_ms = statistics.fmean(latencies) if latencies else 0.0
    metrics.median_latency_ms = statistics.median(latencies) if latencies else 0.0
    metrics.p95_latency_ms = percentile(latencies, 95)
    metrics.llm_calls_per_query = statistics.fmean(r.llm_calls for r in results)
    metrics.token_usage = _sum_tokens(results)
    metrics.confidence_distribution = {
        level.value: sum(1 for result in results if result.confidence == level)
        for level in ConfidenceLevel
    }
    return metrics


def _pairwise(
    results: list[EndToEndCaseResult],
    left: EndToEndSystem,
    right: EndToEndSystem,
) -> PairwiseComparison:
    comparison = PairwiseComparison(left=left, right=right)
    case_ids = sorted({result.case_id for result in results})
    for case_id in case_ids:
        left_result = next(result for result in results if result.case_id == case_id and result.system == left)
        right_result = next(result for result in results if result.case_id == case_id and result.system == right)
        left_score = _quality_score(left_result)
        right_score = _quality_score(right_result)
        if left_score > right_score:
            comparison.wins += 1
        elif left_score < right_score:
            comparison.losses += 1
        else:
            comparison.ties += 1
    return comparison


def _quality_score(result: EndToEndCaseResult) -> tuple[int, int, float, float]:
    if result.answer_status == "not_applicable":
        return (0, 0, 0.0, 0.0)
    correctness = int(result.answer_correct or result.abstention_correct or result.disambiguation_correct)
    citation = int(result.citation_status == "valid" and result.trusted_citations > 0)
    return (correctness, citation, result.evidence_recall, -result.latency_ms)


def _agentic_wins(results: list[EndToEndCaseResult]) -> list[str]:
    wins: list[str] = []
    for case_id in sorted({result.case_id for result in results}):
        agentic = next(result for result in results if result.case_id == case_id and result.system == EndToEndSystem.AGENTIC_HYBRID_RAG)
        vector = next(result for result in results if result.case_id == case_id and result.system == EndToEndSystem.VECTOR_RAG)
        hybrid = next(result for result in results if result.case_id == case_id and result.system == EndToEndSystem.HYBRID_RAG)
        if _quality_score(agentic) > max(_quality_score(vector), _quality_score(hybrid)):
            wins.append(case_id)
    return wins


def _agentic_losses(results: list[EndToEndCaseResult]) -> list[str]:
    losses: list[str] = []
    for case_id in sorted({result.case_id for result in results}):
        agentic = next(result for result in results if result.case_id == case_id and result.system == EndToEndSystem.AGENTIC_HYBRID_RAG)
        vector = next(result for result in results if result.case_id == case_id and result.system == EndToEndSystem.VECTOR_RAG)
        hybrid = next(result for result in results if result.case_id == case_id and result.system == EndToEndSystem.HYBRID_RAG)
        if _quality_score(agentic) < max(_quality_score(vector), _quality_score(hybrid)):
            losses.append(case_id)
    return losses


def _failure_analysis(results: list[EndToEndCaseResult]) -> dict[str, dict[str, int]]:
    buckets: dict[str, dict[str, int]] = {}
    for result in results:
        reason = result.failure_reason or "none"
        buckets.setdefault(result.system.value, {})
        buckets[result.system.value][reason] = buckets[result.system.value].get(reason, 0) + 1
    return buckets


def _citation_target_ids(citations: list[Any]) -> set[str]:
    ids: set[str] = set()
    for citation in citations:
        ids.add(citation.evidence_id)
        ids.update(citation.relationship_ids)
        ids.update(citation.entity_ids)
        ids.update(citation.source_chunk_ids)
        if citation.chunk_id:
            ids.add(citation.chunk_id)
        if citation.paper_id:
            ids.add(citation.paper_id)
    return ids


def _citation_provenance_completeness(citations: list[Any]) -> float:
    if not citations:
        return 0.0
    complete = sum(1 for citation in citations if citation.provenance_complete is not False)
    return complete / len(citations)


def _failure_reason(
    *,
    answerable: bool,
    answer_correct: bool,
    abstention_correct: bool,
    disambiguation_correct: bool,
    citation_status: str | None,
    evidence_recall: float,
    abstained: bool,
) -> str | None:
    if answer_correct or abstention_correct or disambiguation_correct:
        return None
    if citation_status in {"invalid", "no_citations", "partially_valid"}:
        return f"citation_{citation_status}"
    if evidence_recall == 0.0:
        return "retrieval_miss"
    if answerable and abstained:
        return "unnecessary_abstention"
    return "answer_incorrect"


def _agentic_failure_reason(result: Any) -> str | None:
    if result.grounding and result.grounding.reason_codes:
        return result.grounding.reason_codes[0].value
    if result.errors:
        return result.errors[0].node
    return None


def _agentic_llm_calls(result: Any) -> int:
    count = 0
    if result.analysis is not None:
        count += 1
    if result.evidence_assessment is not None and result.evidence_assessment.critic_invoked:
        count += 1
    if result.generated_answer is not None:
        count += 1
    return count


def _token_usage(metadata: dict[str, Any]) -> dict[str, int]:
    usage = metadata.get("token_usage")
    if not isinstance(usage, dict):
        return {}
    return {key: int(value) for key, value in usage.items() if isinstance(value, int)}


def _sum_tokens(results: list[EndToEndCaseResult]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for result in results:
        for key, value in result.token_usage.items():
            totals[key] = totals.get(key, 0) + value
    return totals


def _call_delta(before: Any, after: Any, *, default: int) -> int:
    if isinstance(before, int) and isinstance(after, int):
        return max(0, after - before)
    return default


def _rate(values: list[bool]) -> float:
    return 0.0 if not values else sum(1 for value in values if value) / len(values)


def _planning_correctness(
    *,
    case: EndToEndBenchmarkCase,
    system: EndToEndSystem,
    selected_strategy: str | None,
    planner_intent: str | None,
    graph_operation: str | None,
) -> tuple[bool | None, bool | None]:
    if system != EndToEndSystem.AGENTIC_HYBRID_RAG or case.expected_agentic_plan is None:
        return None, None
    expected = case.expected_agentic_plan
    strategy_correct = expected.strategy is None or selected_strategy == expected.strategy.value
    intent_correct = expected.intent is None or planner_intent == expected.intent
    graph_correct = expected.graph_operation is None or graph_operation == expected.graph_operation.value
    return bool(intent_correct and strategy_correct and graph_correct), bool(strategy_correct)
