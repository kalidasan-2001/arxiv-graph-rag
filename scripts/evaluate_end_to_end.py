"""Run controlled end-to-end RAG evaluation and write reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from app.core.config import Settings
from app.domain.enums import RetrievalStrategy
from app.generation.answer import AnswerContextBuilder, GroundedAnswerGenerator
from app.generation.citations import CitationValidator
from app.generation.grounding import GroundingDecisionService
from app.graph.models import GraphNodeInput, GraphRelationshipInput
from app.graph.neo4j_repository import Neo4jGraphRepository
from app.retrieval.critic import EvidenceCriticService, RetrievalRefinementPlanner
from app.retrieval.evidence import EvidenceProvenanceBridge
from app.retrieval.graph_search import GraphRetrievalService
from app.retrieval.hybrid import EvidenceFusionService, HybridRetrievalService
from app.retrieval.planning import QueryAnalysisService, RetrievalPlanner
from app.retrieval.vector_search import VectorSearchService
from app.retrieval.workflow import RetrievalWorkflowService
from app.storage.qdrant.models import VectorPoint, VectorPointPayload, build_qdrant_point_id
from app.storage.qdrant.qdrant_repository import QdrantVectorRepository
from evaluation.end_to_end_runner import EndToEndEvaluationRunner, load_end_to_end_benchmark
from evaluation.reporting import write_end_to_end_reports


class ControlledEmbeddingProvider:
    @property
    def provider_name(self) -> str:
        return "controlled"

    @property
    def model_name(self) -> str:
        return "prompt20-controlled-4d"

    @property
    def dimension(self) -> int:
        return 4

    @property
    def normalize(self) -> bool:
        return True

    @property
    def provider_version(self) -> str:
        return "v1"

    @property
    def config_fingerprint(self) -> str:
        return "prompt20-controlled-embedding"

    def embed_documents(self, texts) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if any(term in lowered for term in ["dataset", "evalset", "citeset"]):
            return [0.0, 1.0, 0.0, 0.0]
        if any(term in lowered for term in ["method", "approach", "limitation", "problem", "attack"]):
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]


class ControlledLLMProvider:
    def __init__(self) -> None:
        self.last_usage = None
        self.calls = 0
        self.planner_calls = 0
        self.critic_calls = 0
        self.answer_calls = 0

    @property
    def provider_name(self) -> str:
        return "controlled_fake"

    @property
    def model_name(self) -> str:
        return "prompt20-controlled"

    @property
    def provider_version(self) -> str:
        return "v1"

    @property
    def temperature(self) -> float:
        return 0.0

    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model, max_output_tokens=None):
        self.calls += 1
        if response_model.__name__ == "EvidenceAssessment":
            self.critic_calls += 1
            has_refinement_evidence = (
                "chunk:e2e:a-refine" in user_prompt
                or "contrastive retrieval encoder" in user_prompt
            )
            if "Refinement needed" in user_prompt and not has_refinement_evidence:
                return response_model.model_validate(
                    {
                        "sufficient": False,
                        "coverage": "partial",
                        "missing_information": ["refined evidence"],
                        "recommended_refinement_type": "vector_expansion",
                    }
                )
            if "deployment cost" in user_prompt or "privacy budgets" in user_prompt:
                return response_model.model_validate(
                    {
                        "sufficient": False,
                        "coverage": "insufficient",
                        "missing_information": ["requested fact absent"],
                        "recommended_refinement_type": "none",
                    }
                )
            return response_model.model_validate(
                {"sufficient": True, "coverage": "complete", "recommended_refinement_type": "none"}
            )
        if response_model.__name__ == "GeneratedGroundedAnswer":
            self.answer_calls += 1
            return response_model.model_validate(_answer_payload(user_prompt))
        self.planner_calls += 1
        return response_model.model_validate(_analysis_payload(user_prompt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="evaluation/end_to_end_benchmark.json")
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-collection", default="controlled_end_to_end_benchmark")
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--neo4j-username", default="neo4j")
    parser.add_argument("--neo4j-password", required=True)
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--baseline-report", default=None)
    args = parser.parse_args()

    _verify_postgres(args.database_url)
    qdrant_client = QdrantClient(url=args.qdrant_url, timeout=10)
    qdrant_client.get_collections()
    vector_repo = QdrantVectorRepository(qdrant_client, args.qdrant_collection)
    if qdrant_client.collection_exists(args.qdrant_collection):
        qdrant_client.delete_collection(args.qdrant_collection)
    _seed_qdrant(vector_repo)

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_username, args.neo4j_password))
    try:
        with driver.session(database=args.neo4j_database) as session:
            session.run("RETURN 1").consume()
            session.run("MATCH (n) DETACH DELETE n").consume()
        graph_repo = Neo4jGraphRepository(driver, args.neo4j_database)
        _seed_neo4j(graph_repo)
        settings = Settings(
            _env_file=None,
            DATABASE_URL=args.database_url,
            QDRANT_URL=args.qdrant_url,
            QDRANT_COLLECTION=args.qdrant_collection,
            NEO4J_URI=args.neo4j_uri,
            NEO4J_USERNAME=args.neo4j_username,
            NEO4J_PASSWORD=args.neo4j_password,
            NEO4J_DATABASE=args.neo4j_database,
            VECTOR_SEARCH_DEFAULT_TOP_K=2,
            VECTOR_SEARCH_MAX_TOP_K=8,
            GRAPH_DEFAULT_LIMIT=5,
            GRAPH_MAX_LIMIT=10,
            HYBRID_DEFAULT_TOP_K=6,
            HYBRID_MAX_TOP_K=10,
            MAX_RETRIEVAL_ROUNDS=2,
        )
        llm = ControlledLLMProvider()
        retrieval_service = _retrieval_service(settings, vector_repo, graph_repo)
        report = EndToEndEvaluationRunner(
            retrieval_service=retrieval_service,
            answer_context_builder=AnswerContextBuilder(settings=settings),
            answer_generator=GroundedAnswerGenerator(llm, settings=settings),
            citation_validator=CitationValidator(settings=settings),
            grounding_service=GroundingDecisionService(settings=settings),
            agentic_workflow=_agentic_workflow(settings, retrieval_service, graph_repo, llm),
            settings=settings,
            metadata_overrides={
                "controlled_environment": "real PostgreSQL, real Qdrant, real Neo4j, deterministic fake LLM/embeddings",
                "qdrant_collection": args.qdrant_collection,
            },
        ).run_dataset(load_end_to_end_benchmark(args.benchmark))
        if args.baseline_report:
            baseline_report = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
            report.metadata["prompt20_baseline_metrics"] = _agentic_delta_payload(baseline_report, report)
        json_path, markdown_path = write_end_to_end_reports(report, args.output_dir)
        print(f"Wrote {json_path}")
        print(f"Wrote {markdown_path}")
    finally:
        with driver.session(database=args.neo4j_database) as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
        driver.close()
        if qdrant_client.collection_exists(args.qdrant_collection):
            qdrant_client.delete_collection(args.qdrant_collection)
        qdrant_client.close()


def _retrieval_service(settings: Settings, vector_repo: QdrantVectorRepository, graph_repo: Neo4jGraphRepository) -> HybridRetrievalService:
    return HybridRetrievalService(
        vector_service=VectorSearchService(
            ControlledEmbeddingProvider(),
            vector_repo,
            default_top_k=settings.VECTOR_SEARCH_DEFAULT_TOP_K,
            max_top_k=settings.VECTOR_SEARCH_MAX_TOP_K,
        ),
        graph_service=GraphRetrievalService(
            graph_repo,
            max_depth=settings.GRAPH_MAX_DEPTH,
            default_limit=settings.GRAPH_DEFAULT_LIMIT,
            max_limit=settings.GRAPH_MAX_LIMIT,
        ),
        provenance_bridge=EvidenceProvenanceBridge(vector_repo, max_supporting_chunks=settings.EVIDENCE_MAX_SUPPORTING_CHUNKS),
        fusion_service=EvidenceFusionService(rrf_k=settings.HYBRID_RRF_K),
        default_top_k=settings.HYBRID_DEFAULT_TOP_K,
        max_top_k=settings.HYBRID_MAX_TOP_K,
    )


def _agentic_workflow(settings: Settings, retrieval_service: HybridRetrievalService, graph_repo: Neo4jGraphRepository, llm: ControlledLLMProvider) -> RetrievalWorkflowService:
    return RetrievalWorkflowService(
        analysis_service=QueryAnalysisService(llm, settings=settings),
        planner=RetrievalPlanner(graph_repo, settings=settings),
        retrieval_service=retrieval_service,
        critic_service=EvidenceCriticService(llm, settings=settings),
        refinement_planner=RetrievalRefinementPlanner(settings=settings),
        settings=settings,
        answer_context_builder=AnswerContextBuilder(settings=settings),
        answer_generator=GroundedAnswerGenerator(llm, settings=settings),
        citation_validator=CitationValidator(settings=settings),
        grounding_decision_service=GroundingDecisionService(settings=settings),
        enable_answer_generation=True,
    )


def _verify_postgres(database_url: str) -> None:
    engine = create_engine(database_url, pool_pre_ping=True, future=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    finally:
        engine.dispose()


def _seed_qdrant(vector_repo: QdrantVectorRepository) -> None:
    vector_repo.ensure_collection(dimension=4, distance="cosine")
    vector_repo.upsert_chunks(
        [
            _point("chunk:e2e:a-method", "paper:arxiv:e2e-a", [1.0, 0.0, 0.0, 0.0], "Controlled Paper A uses GraphRAG for graph reconstruction attacks."),
            _point("chunk:e2e:a-limit", "paper:arxiv:e2e-a", [1.0, 0.0, 0.0, 0.0], "Controlled Paper A reports sparse citations as a limitation."),
            _point("chunk:e2e:a-refine", "paper:arxiv:e2e-a", [0.8, 0.0, 0.2, 0.0], "Refinement needed evidence says Controlled Paper A uses a contrastive retrieval encoder."),
            _point("chunk:e2e:b-problem", "paper:arxiv:e2e-b", [0.6, 0.0, 0.4, 0.0], "Controlled Paper B solves retrieval drift in graph augmented search."),
            _point("chunk:e2e:c-approach", "paper:arxiv:e2e-c", [0.6, 0.0, 0.4, 0.0], "Controlled Paper C proposes a citation-aware GraphRAG approach."),
            _point("chunk:e2e:a-dataset", "paper:arxiv:e2e-a", [0.0, 1.0, 0.0, 0.0], "Controlled Paper A evaluates on EvalSet."),
            _point("chunk:e2e:b-dataset", "paper:arxiv:e2e-b", [0.0, 1.0, 0.0, 0.0], "Controlled Paper B evaluates on EvalSet."),
            _point("chunk:e2e:c-dataset", "paper:arxiv:e2e-c", [0.0, 1.0, 0.0, 0.0], "Controlled Paper C evaluates on CiteSet."),
        ]
    )


def _seed_neo4j(graph_repo: Neo4jGraphRepository) -> None:
    graph_repo.ensure_schema()
    graph_repo.upsert_entities(
        [
            GraphNodeInput(entity_id="paper:arxiv:e2e-a", entity_type="paper", canonical_name="Controlled Paper A"),
            GraphNodeInput(entity_id="paper:arxiv:e2e-b", entity_type="paper", canonical_name="Controlled Paper B"),
            GraphNodeInput(entity_id="paper:arxiv:e2e-c", entity_type="paper", canonical_name="Controlled Paper C"),
            GraphNodeInput(entity_id="paper:arxiv:e2e-amb-1", entity_type="paper", canonical_name="Ambiguous Paper"),
            GraphNodeInput(entity_id="paper:arxiv:e2e-amb-2", entity_type="paper", canonical_name="Ambiguous Paper"),
            GraphNodeInput(entity_id="entity:method:e2e-graphrag", entity_type="method", canonical_name="GraphRAG"),
            GraphNodeInput(entity_id="entity:method:e2e-citationrag", entity_type="method", canonical_name="CitationRAG"),
            GraphNodeInput(entity_id="entity:method:e2e-amb-1", entity_type="method", canonical_name="Ambiguous Method"),
            GraphNodeInput(entity_id="entity:method:e2e-amb-2", entity_type="method", canonical_name="Ambiguous Method"),
            GraphNodeInput(entity_id="entity:dataset:e2e-evalset", entity_type="dataset", canonical_name="EvalSet"),
            GraphNodeInput(entity_id="entity:dataset:e2e-citeset", entity_type="dataset", canonical_name="CiteSet"),
            GraphNodeInput(entity_id="entity:dataset:e2e-amb-1", entity_type="dataset", canonical_name="Ambiguous Dataset"),
            GraphNodeInput(entity_id="entity:dataset:e2e-amb-2", entity_type="dataset", canonical_name="Ambiguous Dataset"),
            GraphNodeInput(entity_id="entity:task:e2e-reconstruction", entity_type="task", canonical_name="graph reconstruction"),
            GraphNodeInput(entity_id="entity:task:e2e-amb-1", entity_type="task", canonical_name="Ambiguous Task"),
            GraphNodeInput(entity_id="entity:task:e2e-amb-2", entity_type="task", canonical_name="Ambiguous Task"),
        ]
    )
    graph_repo.upsert_relationships(
        [
            _rel("rel:e2e:a-method", "paper:arxiv:e2e-a", "entity:method:e2e-graphrag", "uses_method", "chunk:e2e:a-method"),
            _rel("rel:e2e:a-dataset", "paper:arxiv:e2e-a", "entity:dataset:e2e-evalset", "evaluated_on", "chunk:e2e:a-dataset"),
            _rel("rel:e2e:a-task", "paper:arxiv:e2e-a", "entity:task:e2e-reconstruction", "addresses", "chunk:e2e:a-method"),
            _rel("rel:e2e:b-dataset", "paper:arxiv:e2e-b", "entity:dataset:e2e-evalset", "evaluated_on", "chunk:e2e:b-dataset"),
            _rel("rel:e2e:b-cites-a", "paper:arxiv:e2e-b", "paper:arxiv:e2e-a", "cites", "chunk:e2e:b-problem"),
            _rel("rel:e2e:c-method", "paper:arxiv:e2e-c", "entity:method:e2e-citationrag", "uses_method", "chunk:e2e:c-approach"),
            _rel("rel:e2e:c-dataset", "paper:arxiv:e2e-c", "entity:dataset:e2e-citeset", "evaluated_on", "chunk:e2e:c-dataset"),
            _rel("rel:e2e:c-task", "paper:arxiv:e2e-c", "entity:task:e2e-reconstruction", "addresses", "chunk:e2e:c-approach"),
        ]
    )


def _analysis_payload(prompt: str) -> dict[str, Any]:
    query = _query_from_prompt(prompt)
    lowered = query.lower()
    intent = "semantic_explanation"
    entity = "Controlled Paper A"
    entity_type = "paper"
    if "ambiguous method" in lowered:
        intent, entity, entity_type = "papers_for_method", "Ambiguous Method", "method"
    elif "ambiguous dataset" in lowered:
        intent, entity, entity_type = "papers_for_dataset", "Ambiguous Dataset", "dataset"
    elif "ambiguous task" in lowered:
        intent, entity, entity_type = "papers_for_task", "Ambiguous Task", "task"
    elif "ambiguous paper" in lowered:
        intent, entity = "paper_datasets", "Ambiguous Paper"
    elif "methods are used by papers evaluated on" in lowered:
        intent, entity, entity_type = "methods_for_dataset", "EvalSet" if "evalset" in lowered else "CiteSet", "dataset"
    elif "papers citing" in lowered:
        intent, entity = "datasets_from_citing_papers", "Controlled Paper A"
    elif "same dataset" in lowered or "share evalset" in lowered:
        intent, entity = "shared_datasets", "Controlled Paper A" if "paper a" in lowered else "Controlled Paper B"
    elif "same method" in lowered:
        intent, entity = "shared_methods", "Controlled Paper A"
    elif "same task" in lowered:
        intent, entity, entity_type = "papers_for_task", "graph reconstruction", "task"
    elif "citation neighborhood" in lowered:
        intent, entity = "citation_neighborhood", "Controlled Paper A"
    elif "missingset" in lowered:
        intent, entity, entity_type = "papers_for_dataset", "MissingSet", "dataset"
    elif "datasets" in lowered and "explain" not in lowered and "describe" not in lowered:
        intent, entity = "paper_datasets", "Controlled Paper B" if "paper b" in lowered else "Controlled Paper A"
    elif "methods" in lowered and "describe" not in lowered:
        intent, entity = "paper_methods", "Controlled Paper C" if "paper c" in lowered else "Controlled Paper A"
    elif "task" in lowered:
        intent, entity = "paper_tasks", "Controlled Paper A"
    elif "mixed" in lowered or ("explain" in lowered and "dataset" in lowered) or ("describe" in lowered and "dataset" in lowered):
        intent, entity = "mixed_semantic_structural", "Controlled Paper C" if "paper c" in lowered else "Controlled Paper A"
    elif "paper z" in lowered:
        intent, entity = "paper_authors", "Controlled Paper Z"
    return {
        "query": query,
        "intent": intent,
        "semantic_retrieval_required": intent in {"semantic_explanation", "mixed_semantic_structural"},
        "structural_retrieval_required": intent != "semantic_explanation",
        "entities": [{"text": entity, "entity_type": entity_type}],
        "planning_confidence": 0.9,
    }


def _query_from_prompt(prompt: str) -> str:
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt
    query = payload.get("query")
    return str(query) if query is not None else prompt


def _answer_payload(prompt: str) -> dict[str, Any]:
    if "allowed_evidence_markers" not in prompt:
        return {"text": "The available evidence is insufficient."}
    markers = [f"E{i}" for i in range(1, 7) if f'"E{i}"' in prompt]
    if not markers:
        return {"text": "The available evidence is insufficient."}
    text = "The controlled evidence supports GraphRAG, EvalSet, CitationRAG, sparse citations, retrieval drift, contrastive retrieval, CiteSet, and Controlled Paper B "
    text += " ".join(f"[{marker}]" for marker in markers[:2]) + "."
    return {"text": text, "used_evidence_markers": markers[:2]}


def _agentic_delta_payload(baseline_report: dict[str, Any], current_report) -> dict[str, Any]:
    baseline = baseline_report["overall"]["AGENTIC_HYBRID_RAG"]
    current = current_report.overall["AGENTIC_HYBRID_RAG"].model_dump(mode="json")
    metrics = [
        "answer_accuracy",
        "correct_abstention_rate",
        "false_answer_rate",
        "evidence_recall",
        "refinement_rate",
        "refinement_success_rate",
        "high_confidence_error_rate",
        "p95_latency_ms",
    ]
    delta = {
        metric: {
            "prompt20": baseline.get(metric),
            "prompt20_1": current.get(metric),
            "delta": None
            if baseline.get(metric) is None or current.get(metric) is None
            else current[metric] - baseline[metric],
        }
        for metric in metrics
    }
    baseline_high = baseline.get("confidence_distribution", {}).get("high", 0)
    current_high = current.get("confidence_distribution", {}).get("high", 0)
    delta["HIGH_count"] = {
        "prompt20": baseline_high,
        "prompt20_1": current_high,
        "delta": current_high - baseline_high,
    }
    return {
        "baseline_run_fingerprint": baseline_report.get("run_fingerprint"),
        "delta": delta,
    }


def _point(chunk_id: str, paper_id: str, vector: list[float], text: str) -> VectorPoint:
    return VectorPoint(point_id=build_qdrant_point_id(chunk_id), vector=vector, payload=_payload(chunk_id, paper_id, text))


def _payload(chunk_id: str, paper_id: str, text_value: str) -> VectorPointPayload:
    return VectorPointPayload(
        chunk_id=chunk_id,
        paper_id=paper_id,
        paper_version_id=f"{paper_id}:v1",
        section_id=f"section:{chunk_id}",
        section_type="methodology",
        section_title="Controlled",
        chunk_index=0,
        page_start=1,
        page_end=1,
        source="controlled",
        source_id=paper_id,
        published_year=2026,
        categories=["cs.IR"],
        chunking_version="v1",
        chunk_config_fingerprint="chunk-current",
        embedding_provider="controlled",
        embedding_model="controlled",
        embedding_config_fingerprint="prompt20-controlled-embedding",
        vector_generation_fingerprint="vector-current",
        text=text_value,
    )


def _rel(relationship_id: str, source_id: str, target_id: str, relationship_type: str, chunk_id: str) -> GraphRelationshipInput:
    return GraphRelationshipInput(
        relationship_id=relationship_id,
        source_entity_id=source_id,
        target_entity_id=target_id,
        relationship_type=relationship_type,
        confidence=1.0,
        extraction_version="v1",
        source_chunk_id=chunk_id,
        supporting_chunk_ids=[chunk_id],
        provenance_type="chunk",
        paper_version_id=f"{source_id}:v1" if source_id.startswith("paper:") else "paper:arxiv:e2e:v1",
        graph_index_generation_fingerprint="graph-current",
    )


if __name__ == "__main__":
    main()
