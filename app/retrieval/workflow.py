"""Bounded LangGraph orchestration for query planning and retrieval."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings
from app.domain.enums import ConfidenceLevel, EntityType
from app.domain.evidence import EvidenceItem, EvidencePool, build_evidence_pool
from app.domain.ids import ensure_json_safe, normalize_whitespace
from app.domain.retrieval import RetrievalPlan
from app.generation.answer import (
    AnswerContextBuilder,
    AnswerGenerationContext,
    GeneratedGroundedAnswer,
    GroundedAnswerGenerator,
    answer_generation_config_fingerprint,
)
from app.generation.citations import (
    CitationValidationResult,
    CitationValidationStatus,
    CitationValidator,
    ValidatedGroundedAnswer,
)
from app.generation.grounding import (
    FinalAnswerStatus,
    FinalResearchAnswer,
    GroundingDecision,
    GroundingDecisionService,
)
from app.retrieval.critic import (
    EvidenceAssessment,
    EvidenceCriticService,
    EvidenceRoundSummary,
    RefinementType,
    RetrievalRefinement,
    RetrievalRefinementPlanner,
)
from app.retrieval.graph_search import GraphSearchOperation
from app.retrieval.hybrid import HybridRetrievalResult, HybridRetrievalService
from app.retrieval.planning import (
    AmbiguousEntity,
    EntityResolutionResult,
    PlanningStatus,
    QueryAnalysisService,
    QueryPlanningResult,
    ResolvedEntity,
    RetrievalPlanner,
    StructuredQueryAnalysis,
)

logger = logging.getLogger(__name__)


class RetrievalWorkflowStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PLANNING_FAILED = "PLANNING_FAILED"
    REQUIRES_DISAMBIGUATION = "REQUIRES_DISAMBIGUATION"
    ENTITY_NOT_FOUND = "ENTITY_NOT_FOUND"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    EMPTY_EVIDENCE = "EMPTY_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REFINEMENT_FAILED = "REFINEMENT_FAILED"
    CRITIC_FAILED = "CRITIC_FAILED"
    ANSWER_GENERATION_FAILED = "ANSWER_GENERATION_FAILED"
    CITATION_VALIDATION_FAILED = "CITATION_VALIDATION_FAILED"


class WorkflowError(BaseModel):
    node: str
    error_type: str
    message: str


class WorkflowTraceEvent(BaseModel):
    node: str
    status: str
    duration_ms: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)


class RetrievalWorkflowResult(BaseModel):
    query: str
    status: RetrievalWorkflowStatus
    analysis: StructuredQueryAnalysis | None = None
    planning_status: PlanningStatus | None = None
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    ambiguous_entities: list[AmbiguousEntity] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan | None = None
    retrieval_result: HybridRetrievalResult | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    evidence_pool: EvidencePool | None = None
    evidence_assessment: EvidenceAssessment | None = None
    retrieval_round: int = 0
    refinement: RetrievalRefinement | None = None
    evidence_history: list[EvidenceRoundSummary] = Field(default_factory=list)
    evidence_sufficient: bool | None = None
    missing_information: list[str] = Field(default_factory=list)
    refinement_reason: str | None = None
    answer_context: AnswerGenerationContext | None = None
    generated_answer: GeneratedGroundedAnswer | None = None
    validated_answer: ValidatedGroundedAnswer | None = None
    answer: str | None = None
    citations: list[Any] = Field(default_factory=list)
    citation_validation: CitationValidationResult | None = None
    grounding: GroundingDecision | None = None
    final_answer: FinalResearchAnswer | None = None
    final_status: FinalAnswerStatus | None = None
    confidence: ConfidenceLevel | None = None
    finalization_fingerprint: str | None = None
    answer_generation_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[WorkflowError] = Field(default_factory=list)
    trace: list[WorkflowTraceEvent] = Field(default_factory=list)
    timings: dict[str, int] = Field(default_factory=dict)


class RetrievalWorkflowState(TypedDict, total=False):
    query: str
    status: RetrievalWorkflowStatus
    analysis: StructuredQueryAnalysis | None
    resolution: EntityResolutionResult | None
    planning_result: QueryPlanningResult | None
    retrieval_plan: RetrievalPlan | None
    retrieval_result: HybridRetrievalResult | None
    evidence: list[EvidenceItem]
    evidence_pool: EvidencePool | None
    evidence_assessment: EvidenceAssessment | None
    retrieval_round: int
    refinement: RetrievalRefinement | None
    refinement_evidence: list[EvidenceItem]
    evidence_history: list[EvidenceRoundSummary]
    evidence_sufficient: bool | None
    missing_information: list[str]
    refinement_reason: str | None
    answer_context: AnswerGenerationContext | None
    generated_answer: GeneratedGroundedAnswer | None
    validated_answer: ValidatedGroundedAnswer | None
    answer: str | None
    citations: list[Any]
    citation_validation: CitationValidationResult | None
    grounding: GroundingDecision | None
    final_answer: FinalResearchAnswer | None
    final_status: FinalAnswerStatus | None
    confidence: ConfidenceLevel | None
    finalization_fingerprint: str | None
    answer_generation_metadata: dict[str, Any]
    warnings: list[str]
    errors: list[WorkflowError]
    trace: list[WorkflowTraceEvent]
    timings: dict[str, int]


class RetrievalWorkflowService:
    """Run one short, synchronous LangGraph retrieval workflow."""

    def __init__(
        self,
        *,
        analysis_service: QueryAnalysisService,
        planner: RetrievalPlanner,
        retrieval_service: HybridRetrievalService,
        critic_service: EvidenceCriticService,
        refinement_planner: RetrievalRefinementPlanner,
        settings: Settings,
        answer_context_builder: AnswerContextBuilder | None = None,
        answer_generator: GroundedAnswerGenerator | None = None,
        citation_validator: CitationValidator | None = None,
        grounding_decision_service: GroundingDecisionService | None = None,
        enable_answer_generation: bool = False,
    ) -> None:
        self._analysis_service = analysis_service
        self._planner = planner
        self._retrieval_service = retrieval_service
        self._critic_service = critic_service
        self._refinement_planner = refinement_planner
        self._settings = settings
        self._answer_context_builder = answer_context_builder
        self._answer_generator = answer_generator
        self._citation_validator = citation_validator
        self._grounding_decision_service = grounding_decision_service or GroundingDecisionService(settings=settings)
        self._enable_answer_generation = enable_answer_generation
        self._graph = self._build_graph()

    def run(self, query: str) -> RetrievalWorkflowResult:
        normalized_query = normalize_whitespace(query)
        if not normalized_query:
            return RetrievalWorkflowResult(
                query="",
                status=RetrievalWorkflowStatus.PLANNING_FAILED,
                errors=[
                    WorkflowError(
                        node="input",
                        error_type="ValueError",
                        message="query must not be blank",
                    )
                ],
            )

        initial: RetrievalWorkflowState = {
            "query": normalized_query,
            "warnings": [],
            "errors": [],
            "trace": [],
            "timings": {},
            "evidence": [],
            "retrieval_round": 0,
            "refinement_evidence": [],
            "evidence_history": [],
            "missing_information": [],
            "answer_generation_metadata": {},
            "citations": [],
        }
        final_state = self._graph.invoke(initial)
        result = self._result_from_state(final_state)
        logger.info(
            "retrieval workflow status=%s intent=%s strategy=%s graph_operation=%s "
            "retrieval_rounds=%d evidence_count=%d critic_invoked=%s refinement_type=%s "
            "total_duration_ms=%d",
            result.status.value,
            result.analysis.intent.value if result.analysis else None,
            result.retrieval_plan.strategy.value if result.retrieval_plan else None,
            result.retrieval_plan.graph_operation if result.retrieval_plan else None,
            result.retrieval_round,
            len(result.evidence),
            result.evidence_assessment.critic_invoked if result.evidence_assessment else False,
            result.refinement.refinement_type.value if result.refinement else None,
            result.timings.get("total", 0),
        )
        return result

    def _build_graph(self):
        graph = StateGraph(RetrievalWorkflowState)
        graph.add_node("analyze_query", self._analyze_query)
        graph.add_node("resolve_entities", self._resolve_entities)
        graph.add_node("build_plan", self._build_plan)
        graph.add_node("execute_retrieval", self._execute_retrieval)
        graph.add_node("build_evidence_pool", self._build_evidence_pool)
        graph.add_node("evaluate_evidence", self._evaluate_evidence)
        graph.add_node("build_refinement", self._build_refinement)
        graph.add_node("execute_refinement", self._execute_refinement)
        graph.add_node("merge_evidence", self._merge_evidence)
        graph.add_node("prepare_answer_context", self._prepare_answer_context)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("validate_citations", self._validate_citations)
        graph.add_node("finalize_answer", self._finalize_answer)

        graph.add_edge(START, "analyze_query")
        graph.add_conditional_edges(
            "analyze_query",
            self._route_after_analyze,
            {"continue": "resolve_entities", "finalize": "finalize_answer", "end": END},
        )
        graph.add_edge("resolve_entities", "build_plan")
        graph.add_conditional_edges(
            "build_plan",
            self._route_after_plan,
            {"execute": "execute_retrieval", "finalize": "finalize_answer", "end": END},
        )
        graph.add_conditional_edges(
            "execute_retrieval",
            self._route_after_retrieval,
            {"pool": "build_evidence_pool", "finalize": "finalize_answer", "end": END},
        )
        graph.add_edge("build_evidence_pool", "evaluate_evidence")
        graph.add_conditional_edges(
            "evaluate_evidence",
            self._route_after_evaluation,
            {"answer": "prepare_answer_context", "refine": "build_refinement", "finalize": "finalize_answer", "end": END},
        )
        graph.add_conditional_edges(
            "prepare_answer_context",
            self._route_after_prepare_answer_context,
            {"generate": "generate_answer", "finalize": "finalize_answer", "end": END},
        )
        graph.add_conditional_edges(
            "generate_answer",
            self._route_after_generate_answer,
            {"validate": "validate_citations", "finalize": "finalize_answer", "end": END},
        )
        graph.add_edge("validate_citations", "finalize_answer")
        graph.add_conditional_edges(
            "build_refinement",
            self._route_after_build_refinement,
            {"execute": "execute_refinement", "finalize": "finalize_answer", "end": END},
        )
        graph.add_conditional_edges(
            "execute_refinement",
            self._route_after_refinement,
            {"merge": "merge_evidence", "finalize": "finalize_answer", "end": END},
        )
        graph.add_edge("merge_evidence", "build_evidence_pool")
        graph.add_edge("finalize_answer", END)
        return graph.compile()

    def _analyze_query(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        try:
            analysis = self._analysis_service.analyze(state["query"])
            duration = _duration_ms(started)
            return _update_state(
                state,
                analysis=analysis,
                trace=_append_trace(
                    state,
                    "analyze_query",
                    "ok",
                    duration,
                    {"intent": analysis.intent.value, "entity_mentions": len(analysis.entities)},
                ),
                timings=_add_timing(state, "planning", duration),
            )
        except Exception as exc:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.PLANNING_FAILED,
                errors=_append_error(state, "analyze_query", exc),
                trace=_append_trace(state, "analyze_query", "error", duration, {"error_type": type(exc).__name__}),
                timings=_add_timing(state, "planning", duration),
            )

    def _resolve_entities(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        analysis = state.get("analysis")
        if analysis is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.PLANNING_FAILED,
                trace=_append_trace(state, "resolve_entities", "skipped", duration, {}),
            )
        resolution = self._planner.resolve_entities(analysis)
        duration = _duration_ms(started)
        status = "ok"
        if resolution.ambiguous_entities:
            status = "ambiguous"
        elif resolution.missing_entities:
            status = "entity_not_found"
        return _update_state(
            state,
            resolution=resolution,
            trace=_append_trace(
                state,
                "resolve_entities",
                status,
                duration,
                {
                    "resolved_entities": len(resolution.resolved_entities),
                    "ambiguous_entities": len(resolution.ambiguous_entities),
                    "missing_entities": len(resolution.missing_entities),
                },
            ),
            timings=_add_timing(state, "entity_resolution", duration),
        )

    def _build_plan(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        analysis = state.get("analysis")
        if analysis is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.PLANNING_FAILED,
                trace=_append_trace(state, "build_plan", "skipped", duration, {}),
            )
        planning_result = self._planner.build_plan(analysis, state.get("resolution"))
        workflow_status = _workflow_status_for_planning(planning_result.status)
        duration = _duration_ms(started)
        updates: dict[str, Any] = {
            "planning_result": planning_result,
            "retrieval_plan": planning_result.plan,
            "trace": _append_trace(
                state,
                "build_plan",
                "ok" if planning_result.status == PlanningStatus.OK else planning_result.status.value,
                duration,
                {
                    "planning_status": planning_result.status.value,
                    "strategy": planning_result.plan.strategy.value if planning_result.plan else None,
                    "graph_operation": planning_result.plan.graph_operation if planning_result.plan else None,
                },
            ),
            "timings": _add_timing(state, "planning", duration),
        }
        if planning_result.status != PlanningStatus.OK:
            updates["status"] = workflow_status
            if planning_result.failure_reason:
                updates["errors"] = [
                    *state.get("errors", []),
                    WorkflowError(
                        node="build_plan",
                        error_type=planning_result.status.value,
                        message=planning_result.failure_reason,
                    ),
                ]
        return _update_state(state, **updates)

    def _execute_retrieval(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        plan = state.get("retrieval_plan")
        if plan is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.PLANNING_FAILED,
                trace=_append_trace(state, "execute_retrieval", "skipped", duration, {}),
            )
        try:
            graph_request = plan.graph_request or {}
            result = self._retrieval_service.retrieve(
                query=plan.query,
                strategy=plan.strategy,
                top_k=plan.final_top_k,
                vector_top_k=plan.vector_top_k,
                graph_operation=GraphSearchOperation(plan.graph_operation) if plan.graph_operation else None,
                entity_id=graph_request.get("entity_id"),
                entity_type=EntityType(graph_request["entity_type"]) if graph_request.get("entity_type") else None,
                canonical_name=graph_request.get("canonical_name"),
                graph_depth=graph_request.get("depth"),
                graph_limit=plan.graph_limit,
                paper_id=plan.filters.get("paper_id") if plan.strategy.value in {"vector", "hybrid"} else None,
            )
            evidence = [item.evidence for item in result.evidence]
            duration = _duration_ms(started)
            return _update_state(
                state,
                retrieval_result=result,
                evidence=evidence,
                retrieval_round=1,
                warnings=[*state.get("warnings", []), *result.warnings],
                evidence_history=[
                    *state.get("evidence_history", []),
                    EvidenceRoundSummary(
                        retrieval_round=1,
                        strategy=plan.strategy,
                        evidence_count=len(evidence),
                        new_unique_evidence_count=len({item.evidence_id for item in evidence}),
                    ),
                ],
                trace=_append_trace(
                    state,
                    "execute_retrieval",
                    "ok",
                    duration,
                    {
                        "strategy": plan.strategy.value,
                        "evidence_count": len(evidence),
                        "vector_candidates": result.diagnostics.get("vector_candidates"),
                        "graph_candidates": result.diagnostics.get("graph_candidates"),
                    },
                ),
                timings=_add_timing(state, "retrieval", duration),
            )
        except Exception as exc:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.RETRIEVAL_FAILED,
                errors=_append_error(state, "execute_retrieval", exc),
                trace=_append_trace(
                    state,
                    "execute_retrieval",
                    "error",
                    duration,
                    {"strategy": plan.strategy.value, "error_type": type(exc).__name__},
                ),
                timings=_add_timing(state, "retrieval", duration),
            )

    def _build_evidence_pool(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        evidence = state.get("evidence", [])
        evidence_pool = build_evidence_pool(evidence)
        duration = _duration_ms(started)
        status = state.get("status")
        if status in {
            RetrievalWorkflowStatus.REFINEMENT_FAILED,
            RetrievalWorkflowStatus.RETRIEVAL_FAILED,
            RetrievalWorkflowStatus.CRITIC_FAILED,
        }:
            resolved_status = status
        else:
            resolved_status = RetrievalWorkflowStatus.SUCCESS if evidence else RetrievalWorkflowStatus.EMPTY_EVIDENCE
        return _update_state(
            state,
            status=resolved_status,
            evidence_pool=evidence_pool,
            trace=_append_trace(
                state,
                "build_evidence_pool",
                "ok",
                duration,
                {"pool_size": len(evidence_pool.items), "evidence_count": len(evidence)},
            ),
            timings=_add_timing(state, "evidence_pool", duration),
        )

    def _evaluate_evidence(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        analysis = state.get("analysis")
        plan = state.get("retrieval_plan")
        evidence_pool = state.get("evidence_pool")
        planning_result = state.get("planning_result")
        if analysis is None or plan is None or evidence_pool is None or planning_result is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.CRITIC_FAILED,
                trace=_append_trace(state, "evaluate_evidence", "skipped", duration, {}),
            )
        try:
            assessment = self._critic_service.assess(
                query=state["query"],
                analysis=analysis,
                plan=plan,
                evidence=state.get("evidence", []),
                evidence_pool=evidence_pool,
                resolved_entities=planning_result.resolved_entities,
            )
            duration = _duration_ms(started)
            round_number = state.get("retrieval_round", 0)
            status = _status_after_assessment(
                current_status=state.get("status"),
                assessment=assessment,
                retrieval_round=round_number,
                max_rounds=self._settings.MAX_RETRIEVAL_ROUNDS,
            )
            return _update_state(
                state,
                status=status,
                evidence_assessment=assessment,
                evidence_sufficient=assessment.sufficient,
                missing_information=assessment.missing_information,
                evidence_history=_mark_latest_round_sufficiency(
                    state.get("evidence_history", []), assessment.sufficient
                ),
                trace=_append_trace(
                    state,
                    "evaluate_evidence",
                    "sufficient" if assessment.sufficient else "insufficient",
                    duration,
                    {
                        "retrieval_round": round_number,
                        "sufficient": assessment.sufficient,
                        "coverage": assessment.coverage.value,
                        "missing_information_count": len(assessment.missing_information),
                        "critic_invoked": assessment.critic_invoked,
                    },
                ),
                timings=_add_timing(state, "critic", duration),
            )
        except Exception as exc:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.CRITIC_FAILED,
                errors=_append_error(state, "evaluate_evidence", exc),
                trace=_append_trace(
                    state,
                    "evaluate_evidence",
                    "error",
                    duration,
                    {"retrieval_round": state.get("retrieval_round", 0), "error_type": type(exc).__name__},
                ),
                timings=_add_timing(state, "critic", duration),
            )

    def _build_refinement(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        plan = state.get("retrieval_plan")
        assessment = state.get("evidence_assessment")
        if plan is None or assessment is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.INSUFFICIENT_EVIDENCE,
                trace=_append_trace(state, "build_refinement", "skipped", duration, {}),
            )
        refinement = self._refinement_planner.build_refinement(
            original_plan=plan,
            assessment=assessment,
            retrieval_round=state.get("retrieval_round", 0),
        )
        duration = _duration_ms(started)
        if refinement is None:
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.INSUFFICIENT_EVIDENCE,
                refinement_reason="no valid refinement available",
                trace=_append_trace(
                    state,
                    "build_refinement",
                    "none",
                    duration,
                    {
                        "retrieval_round": state.get("retrieval_round", 0),
                        "requested_refinement": assessment.recommended_refinement_type.value
                        if assessment.recommended_refinement_type
                        else None,
                    },
                ),
            )
        return _update_state(
            state,
            refinement=refinement,
            refinement_reason=refinement.reason_code,
            trace=_append_trace(
                state,
                "build_refinement",
                "ok",
                duration,
                {
                    "retrieval_round": refinement.retrieval_round,
                    "refinement_type": refinement.refinement_type.value,
                },
            ),
        )

    def _execute_refinement(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        plan = state.get("retrieval_plan")
        refinement = state.get("refinement")
        if plan is None or refinement is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.INSUFFICIENT_EVIDENCE,
                trace=_append_trace(state, "execute_refinement", "skipped", duration, {}),
            )
        try:
            graph_request = plan.graph_request or {}
            result = self._retrieval_service.retrieve(
                query=plan.query,
                strategy=refinement.strategy,
                top_k=plan.final_top_k,
                vector_top_k=refinement.vector_top_k or plan.vector_top_k,
                graph_operation=GraphSearchOperation(plan.graph_operation) if plan.graph_operation else None,
                entity_id=graph_request.get("entity_id"),
                entity_type=EntityType(graph_request["entity_type"]) if graph_request.get("entity_type") else None,
                canonical_name=graph_request.get("canonical_name"),
                graph_depth=refinement.graph_depth or graph_request.get("depth"),
                graph_limit=refinement.graph_limit or plan.graph_limit,
                paper_id=plan.filters.get("paper_id")
                if refinement.strategy.value in {"vector", "hybrid"}
                else None,
            )
            evidence = [item.evidence for item in result.evidence]
            existing_ids = {item.evidence_id for item in state.get("evidence", [])}
            new_unique_count = sum(1 for item in evidence if item.evidence_id not in existing_ids)
            duration = _duration_ms(started)
            status = state.get("status")
            if new_unique_count == 0:
                status = RetrievalWorkflowStatus.INSUFFICIENT_EVIDENCE
            return _update_state(
                state,
                status=status,
                retrieval_round=refinement.retrieval_round,
                refinement_evidence=evidence,
                retrieval_result=result,
                warnings=[*state.get("warnings", []), *result.warnings],
                evidence_history=[
                    *state.get("evidence_history", []),
                    EvidenceRoundSummary(
                        retrieval_round=refinement.retrieval_round,
                        strategy=refinement.strategy,
                        evidence_count=len(evidence),
                        new_unique_evidence_count=new_unique_count,
                        refinement_type=refinement.refinement_type,
                    ),
                ],
                trace=_append_trace(
                    state,
                    "execute_refinement",
                    "ok" if new_unique_count else "no_new_evidence",
                    duration,
                    {
                        "retrieval_round": refinement.retrieval_round,
                        "refinement_type": refinement.refinement_type.value,
                        "evidence_count": len(evidence),
                        "new_evidence_count": new_unique_count,
                    },
                ),
                timings=_add_timing(state, "refinement", duration),
            )
        except Exception as exc:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.REFINEMENT_FAILED,
                errors=_append_error(state, "execute_refinement", exc),
                trace=_append_trace(
                    state,
                    "execute_refinement",
                    "error",
                    duration,
                    {
                        "retrieval_round": refinement.retrieval_round,
                        "refinement_type": refinement.refinement_type.value,
                        "error_type": type(exc).__name__,
                    },
                ),
                timings=_add_timing(state, "refinement", duration),
            )

    def _merge_evidence(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        merged = _merge_evidence_by_id(state.get("evidence", []), state.get("refinement_evidence", []))
        duration = _duration_ms(started)
        return _update_state(
            state,
            evidence=merged,
            trace=_append_trace(
                state,
                "merge_evidence",
                "ok",
                duration,
                {
                    "retrieval_round": state.get("retrieval_round", 0),
                    "total_evidence": len(merged),
                },
            ),
        )

    def _prepare_answer_context(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        if self._answer_context_builder is None or self._answer_generator is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.ANSWER_GENERATION_FAILED,
                errors=[
                    *state.get("errors", []),
                    WorkflowError(
                        node="prepare_answer_context",
                        error_type="ConfigurationError",
                        message="answer generation services are not configured",
                    ),
                ],
                trace=_append_trace(state, "prepare_answer_context", "error", duration, {}),
            )
        analysis = state.get("analysis")
        evidence_pool = state.get("evidence_pool")
        if not _can_generate_answer(state) or evidence_pool is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                trace=_append_trace(state, "prepare_answer_context", "skipped", duration, {}),
            )
        try:
            fingerprint = answer_generation_config_fingerprint(
                settings=self._settings,
                provider_name=self._answer_generator.provider_name,
                model_name=self._answer_generator.model_name,
                temperature=self._answer_generator.temperature,
            )
            context = self._answer_context_builder.build(
                query=state["query"],
                analysis=analysis,
                evidence_pool=evidence_pool,
                generation_config_fingerprint=fingerprint,
            )
            duration = _duration_ms(started)
            return _update_state(
                state,
                answer_context=context,
                answer_generation_metadata={
                    **state.get("answer_generation_metadata", {}),
                    "generation_config_fingerprint": fingerprint,
                    "context_fingerprint": context.context_fingerprint,
                    "context_builder_version": context.context_builder_version,
                    "citation_validation": "deferred_prompt_18",
                },
                trace=_append_trace(
                    state,
                    "prepare_answer_context",
                    "ok",
                    duration,
                    {
                        "evidence_items_in_context": len(context.evidence_items),
                        "context_size": context.context_chars,
                        "truncated": context.truncated,
                    },
                ),
                timings=_add_timing(state, "answer_context", duration),
            )
        except Exception as exc:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.ANSWER_GENERATION_FAILED,
                errors=_append_error(state, "prepare_answer_context", exc),
                trace=_append_trace(
                    state,
                    "prepare_answer_context",
                    "error",
                    duration,
                    {"error_type": type(exc).__name__},
                ),
                timings=_add_timing(state, "answer_context", duration),
            )

    def _generate_answer(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        if self._answer_generator is None or not _can_generate_answer(state):
            duration = _duration_ms(started)
            return _update_state(
                state,
                trace=_append_trace(state, "generate_answer", "skipped", duration, {}),
            )
        context = state.get("answer_context")
        if context is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.ANSWER_GENERATION_FAILED,
                trace=_append_trace(state, "generate_answer", "skipped", duration, {}),
            )
        try:
            answer = self._answer_generator.generate(context=context)
            duration = _duration_ms(started)
            usage = answer.generation_metadata.get("token_usage")
            return _update_state(
                state,
                generated_answer=answer,
                answer_generation_metadata={
                    **state.get("answer_generation_metadata", {}),
                    **answer.generation_metadata,
                },
                trace=_append_trace(
                    state,
                    "generate_answer",
                    "ok",
                    duration,
                    {
                        "answer_chars": len(answer.text),
                        "model": answer.generation_metadata.get("model"),
                        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                        "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
                        "total_tokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
                    },
                ),
                timings=_add_timing(state, "generation", duration),
            )
        except Exception as exc:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.ANSWER_GENERATION_FAILED,
                errors=_append_error(state, "generate_answer", exc),
                trace=_append_trace(
                    state,
                    "generate_answer",
                    "error",
                    duration,
                    {"error_type": type(exc).__name__},
                ),
                timings=_add_timing(state, "generation", duration),
            )

    def _validate_citations(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        if self._citation_validator is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.CITATION_VALIDATION_FAILED,
                errors=[
                    *state.get("errors", []),
                    WorkflowError(
                        node="validate_citations",
                        error_type="ConfigurationError",
                        message="citation validator is not configured",
                    ),
                ],
                trace=_append_trace(state, "validate_citations", "error", duration, {}),
            )
        generated_answer = state.get("generated_answer")
        evidence_pool = state.get("evidence_pool")
        answer_context = state.get("answer_context")
        if generated_answer is None or evidence_pool is None or answer_context is None:
            duration = _duration_ms(started)
            return _update_state(
                state,
                trace=_append_trace(state, "validate_citations", "skipped", duration, {}),
            )
        try:
            validated = self._citation_validator.validate(
                generated_answer=generated_answer,
                evidence_pool=evidence_pool,
                answer_context=answer_context,
            )
            validation = validated.citation_validation
            warnings = [*state.get("warnings", []), *validation.warnings]
            status = state.get("status")
            if validation.validation_status in {
                CitationValidationStatus.INVALID,
                CitationValidationStatus.NO_CITATIONS,
            }:
                status = RetrievalWorkflowStatus.CITATION_VALIDATION_FAILED
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=status,
                validated_answer=validated,
                answer=validated.text,
                citations=validated.citations,
                citation_validation=validation,
                warnings=warnings,
                answer_generation_metadata={
                    **state.get("answer_generation_metadata", {}),
                    **validated.generation_metadata,
                },
                trace=_append_trace(
                    state,
                    "validate_citations",
                    validation.validation_status.value,
                    duration,
                    {
                        "markers_found": len(validation.valid_markers) + len(validation.invalid_markers),
                        "valid_markers": len(validation.valid_markers),
                        "invalid_markers": len(validation.invalid_markers),
                        "trusted_citations": len(validation.citations),
                        "validation_status": validation.validation_status.value,
                    },
                ),
                timings=_add_timing(state, "citation_validation", duration),
            )
        except Exception as exc:
            duration = _duration_ms(started)
            return _update_state(
                state,
                status=RetrievalWorkflowStatus.CITATION_VALIDATION_FAILED,
                errors=_append_error(state, "validate_citations", exc),
                trace=_append_trace(
                    state,
                    "validate_citations",
                    "error",
                    duration,
                    {"error_type": type(exc).__name__},
                ),
                timings=_add_timing(state, "citation_validation", duration),
            )

    def _finalize_answer(self, state: RetrievalWorkflowState) -> RetrievalWorkflowState:
        started = time.monotonic()
        final_answer = self._grounding_decision_service.decide(
            query=state["query"],
            internal_status=state.get("status", RetrievalWorkflowStatus.PLANNING_FAILED).value,
            evidence=state.get("evidence", []),
            evidence_assessment=state.get("evidence_assessment"),
            citation_validation=state.get("citation_validation"),
            validated_answer=state.get("validated_answer"),
            retrieval_round=state.get("retrieval_round", 0),
            warnings=state.get("warnings", []),
            analysis_intent=state["analysis"].intent.value if state.get("analysis") is not None else None,
        )
        duration = _duration_ms(started)
        return _update_state(
            state,
            final_answer=final_answer,
            final_status=final_answer.status,
            grounding=final_answer.grounding,
            confidence=final_answer.confidence,
            finalization_fingerprint=final_answer.grounding.grounding_fingerprint,
            answer=final_answer.answer,
            citations=final_answer.citations,
            warnings=final_answer.warnings,
            answer_generation_metadata={
                **state.get("answer_generation_metadata", {}),
                "grounding_fingerprint": final_answer.grounding.grounding_fingerprint,
                "grounding_rules_version": self._settings.GROUNDING_RULES_VERSION,
                "confidence_rules_version": self._settings.CONFIDENCE_RULES_VERSION,
                "abstention_template_version": self._settings.ABSTENTION_TEMPLATE_VERSION,
            },
            trace=_append_trace(
                state,
                "finalize_answer",
                final_answer.status.value,
                duration,
                {
                    "allow_answer": final_answer.grounding.allow_answer,
                    "confidence": final_answer.confidence.value,
                    "trusted_citations": len(final_answer.citations),
                    "warning_count": len(final_answer.warnings),
                    "reason_codes": [code.value for code in final_answer.grounding.reason_codes],
                },
            ),
            timings=_add_timing(state, "finalization", duration),
        )

    def _route_after_analyze(self, state: RetrievalWorkflowState) -> str:
        if state.get("status") == RetrievalWorkflowStatus.PLANNING_FAILED:
            return "finalize" if self._enable_answer_generation else "end"
        return "continue"

    def _route_after_plan(self, state: RetrievalWorkflowState) -> str:
        if state.get("retrieval_plan") is not None:
            return "execute"
        return "finalize" if self._enable_answer_generation else "end"

    def _route_after_retrieval(self, state: RetrievalWorkflowState) -> str:
        if state.get("status") == RetrievalWorkflowStatus.RETRIEVAL_FAILED:
            return "finalize" if self._enable_answer_generation else "end"
        return "pool"

    def _route_after_evaluation(self, state: RetrievalWorkflowState) -> str:
        if state.get("status") in {
            RetrievalWorkflowStatus.CRITIC_FAILED,
            RetrievalWorkflowStatus.INSUFFICIENT_EVIDENCE,
        }:
            return "finalize" if self._enable_answer_generation else "end"
        if state.get("status") == RetrievalWorkflowStatus.SUCCESS:
            return "answer" if self._enable_answer_generation else "end"
        return "refine"

    def _route_after_prepare_answer_context(self, state: RetrievalWorkflowState) -> str:
        if state.get("answer_context") is not None:
            return "generate"
        return "finalize" if self._enable_answer_generation else "end"

    def _route_after_generate_answer(self, state: RetrievalWorkflowState) -> str:
        if state.get("status") == RetrievalWorkflowStatus.ANSWER_GENERATION_FAILED:
            return "finalize" if self._enable_answer_generation else "end"
        if state.get("generated_answer") is not None:
            return "validate"
        return "finalize" if self._enable_answer_generation else "end"

    def _route_after_build_refinement(self, state: RetrievalWorkflowState) -> str:
        if state.get("refinement") is not None:
            return "execute"
        return "finalize" if self._enable_answer_generation else "end"

    def _route_after_refinement(self, state: RetrievalWorkflowState) -> str:
        if state.get("status") in {
            RetrievalWorkflowStatus.REFINEMENT_FAILED,
            RetrievalWorkflowStatus.INSUFFICIENT_EVIDENCE,
        }:
            return "finalize" if self._enable_answer_generation else "end"
        return "merge"

    @staticmethod
    def _result_from_state(state: RetrievalWorkflowState) -> RetrievalWorkflowResult:
        timings = dict(state.get("timings", {}))
        timings["total"] = sum(timings.values())
        planning_result = state.get("planning_result")
        return RetrievalWorkflowResult(
            query=state["query"],
            status=state.get("status", RetrievalWorkflowStatus.PLANNING_FAILED),
            analysis=state.get("analysis"),
            planning_status=planning_result.status if planning_result else None,
            resolved_entities=planning_result.resolved_entities if planning_result else [],
            ambiguous_entities=planning_result.ambiguous_entities if planning_result else [],
            retrieval_plan=state.get("retrieval_plan"),
            retrieval_result=state.get("retrieval_result"),
            evidence=state.get("evidence", []),
            evidence_pool=state.get("evidence_pool"),
            evidence_assessment=state.get("evidence_assessment"),
            retrieval_round=state.get("retrieval_round", 0),
            refinement=state.get("refinement"),
            evidence_history=state.get("evidence_history", []),
            evidence_sufficient=state.get("evidence_sufficient"),
            missing_information=state.get("missing_information", []),
            refinement_reason=state.get("refinement_reason"),
            answer_context=state.get("answer_context"),
            generated_answer=state.get("generated_answer"),
            validated_answer=state.get("validated_answer"),
            answer=state.get("answer"),
            citations=state.get("citations", []),
            citation_validation=state.get("citation_validation"),
            grounding=state.get("grounding"),
            final_answer=state.get("final_answer"),
            final_status=state.get("final_status"),
            confidence=state.get("confidence"),
            finalization_fingerprint=state.get("finalization_fingerprint"),
            answer_generation_metadata=state.get("answer_generation_metadata", {}),
            warnings=state.get("warnings", []),
            errors=state.get("errors", []),
            trace=state.get("trace", []),
            timings=timings,
        )


def _workflow_status_for_planning(status: PlanningStatus) -> RetrievalWorkflowStatus:
    if status == PlanningStatus.AMBIGUOUS:
        return RetrievalWorkflowStatus.REQUIRES_DISAMBIGUATION
    if status == PlanningStatus.ENTITY_NOT_FOUND:
        return RetrievalWorkflowStatus.ENTITY_NOT_FOUND
    if status == PlanningStatus.UNSUPPORTED_GRAPH_OPERATION:
        return RetrievalWorkflowStatus.UNSUPPORTED_OPERATION
    return RetrievalWorkflowStatus.PLANNING_FAILED


def _duration_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _update_state(state: RetrievalWorkflowState, **updates: Any) -> RetrievalWorkflowState:
    return {**state, **updates}


def _append_trace(
    state: RetrievalWorkflowState,
    node: str,
    status: str,
    duration_ms: int,
    metadata: dict[str, Any],
) -> list[WorkflowTraceEvent]:
    return [
        *state.get("trace", []),
        WorkflowTraceEvent(node=node, status=status, duration_ms=duration_ms, metadata=metadata),
    ]


def _append_error(
    state: RetrievalWorkflowState,
    node: str,
    exc: Exception,
) -> list[WorkflowError]:
    return [
        *state.get("errors", []),
        WorkflowError(node=node, error_type=type(exc).__name__, message=str(exc)),
    ]


def _add_timing(state: RetrievalWorkflowState, name: str, duration_ms: int) -> dict[str, int]:
    return {**state.get("timings", {}), name: state.get("timings", {}).get(name, 0) + duration_ms}


def _can_generate_answer(state: RetrievalWorkflowState) -> bool:
    return (
        state.get("status") == RetrievalWorkflowStatus.SUCCESS
        and state.get("evidence_sufficient") is True
        and bool(state.get("evidence_pool") and state["evidence_pool"].items)
        and not state.get("errors")
    )


def _status_after_assessment(
    *,
    current_status: RetrievalWorkflowStatus | None,
    assessment: EvidenceAssessment,
    retrieval_round: int,
    max_rounds: int,
) -> RetrievalWorkflowStatus:
    if assessment.sufficient:
        return RetrievalWorkflowStatus.SUCCESS
    if retrieval_round >= max_rounds:
        return RetrievalWorkflowStatus.INSUFFICIENT_EVIDENCE
    if current_status == RetrievalWorkflowStatus.EMPTY_EVIDENCE:
        return RetrievalWorkflowStatus.EMPTY_EVIDENCE
    return RetrievalWorkflowStatus.EMPTY_EVIDENCE if retrieval_round == 0 else RetrievalWorkflowStatus.EMPTY_EVIDENCE


def _mark_latest_round_sufficiency(
    history: list[EvidenceRoundSummary], sufficient: bool
) -> list[EvidenceRoundSummary]:
    if not history:
        return history
    updated = list(history)
    latest = updated[-1]
    updated[-1] = latest.model_copy(update={"sufficient": sufficient})
    return updated


def _merge_evidence_by_id(
    initial: list[EvidenceItem], refinement: list[EvidenceItem]
) -> list[EvidenceItem]:
    merged: dict[str, EvidenceItem] = {}
    for round_number, items in [(1, initial), (2, refinement)]:
        for item in items:
            rounds = list(item.metadata.get("retrieval_rounds", []))
            if round_number not in rounds:
                rounds.append(round_number)
            updated = item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "retrieval_rounds": sorted(rounds),
                        "first_seen_round": item.metadata.get("first_seen_round", round_number),
                    }
                }
            )
            if item.evidence_id in merged:
                existing = merged[item.evidence_id]
                existing_rounds = set(existing.metadata.get("retrieval_rounds", [])) | set(
                    updated.metadata.get("retrieval_rounds", [])
                )
                merged[item.evidence_id] = existing.model_copy(
                    update={
                        "metadata": {
                            **existing.metadata,
                            "retrieval_rounds": sorted(existing_rounds),
                        }
                    }
                )
            else:
                merged[item.evidence_id] = updated
    return list(merged.values())
