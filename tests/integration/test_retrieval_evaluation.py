"""Controlled retrieval evaluation through real Qdrant and Neo4j services."""

from pathlib import Path

from app.core.config import Settings
from app.domain.enums import RetrievalStrategy
from app.graph.models import GraphNodeInput, GraphRelationshipInput
from app.graph.neo4j_repository import Neo4jGraphRepository
from app.retrieval.evidence import EvidenceProvenanceBridge
from app.retrieval.graph_search import GraphRetrievalService
from app.retrieval.hybrid import EvidenceFusionService, HybridRetrievalService
from app.retrieval.vector_search import VectorSearchService
from app.storage.qdrant.models import VectorPoint, VectorPointPayload, build_qdrant_point_id
from app.storage.qdrant.qdrant_repository import QdrantVectorRepository
from evaluation.reporting import write_reports
from evaluation.runner import RetrievalEvaluationRunner, load_benchmark


class _EvalEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if "limitation" in lowered:
            return [0.0, 1.0, 0.0, 0.0]
        if "graph reconstruction" in lowered:
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]


def _payload(chunk_id: str, paper_id: str, text: str) -> VectorPointPayload:
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
        chunking_version="chunk-v1",
        chunk_config_fingerprint="chunk-fp",
        embedding_provider="fake",
        embedding_model="fake-model",
        embedding_config_fingerprint="embed-fp",
        vector_generation_fingerprint="vector-current",
        text=text,
    )


def _point(chunk_id: str, paper_id: str, vector: list[float], text: str) -> VectorPoint:
    return VectorPoint(
        point_id=build_qdrant_point_id(chunk_id),
        vector=vector,
        payload=_payload(chunk_id, paper_id, text),
    )


def test_controlled_retrieval_evaluation_runs_three_strategies(
    qdrant_client,
    qdrant_collection_name,
    neo4j_repository: Neo4jGraphRepository,
    tmp_path,
) -> None:
    vector_repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
    vector_repo.ensure_collection(dimension=4, distance="cosine")
    vector_repo.upsert_chunks(
        [
            _point("chunk:eval:a-method", "paper:arxiv:eval-a", [1.0, 0.01, 0.0, 0.0], "Graph reconstruction attack method."),
            _point("chunk:eval:a-limit", "paper:arxiv:eval-a", [0.02, 1.0, 0.0, 0.0], "The limitation is sparse labels."),
            _point("chunk:eval:a-dataset", "paper:arxiv:eval-a", [0.90, 0.02, 1.0, 0.0], "Paper A uses EvalSet."),
            _point("chunk:eval:a-task", "paper:arxiv:eval-a", [0.70, 0.03, 1.0, 0.0], "Paper A addresses reconstruction."),
            _point("chunk:eval:b-dataset", "paper:arxiv:eval-b", [0.50, 0.04, 1.0, 0.0], "Paper B uses EvalSet."),
            _point("chunk:eval:b-cite", "paper:arxiv:eval-b", [0.30, 0.05, 1.0, 0.0], "Paper B cites Paper A."),
            _point("chunk:eval:c-method", "paper:arxiv:eval-c", [0.10, 0.06, 1.0, 0.0], "Paper C uses GraphRAG."),
            _point("chunk:eval:b-method", "paper:arxiv:eval-b", [0.01, 0.07, 1.0, 0.0], "Paper B uses a baseline method."),
        ]
    )

    neo4j_repository.ensure_schema()
    neo4j_repository.upsert_entities(
        [
            GraphNodeInput(entity_id="paper:arxiv:eval-a", entity_type="paper", canonical_name="Controlled Paper A"),
            GraphNodeInput(entity_id="paper:arxiv:eval-b", entity_type="paper", canonical_name="Controlled Paper B"),
            GraphNodeInput(entity_id="paper:arxiv:eval-c", entity_type="paper", canonical_name="Controlled Paper C"),
            GraphNodeInput(entity_id="entity:dataset:eval-shared", entity_type="dataset", canonical_name="EvalSet"),
            GraphNodeInput(entity_id="entity:method:eval-graphrag", entity_type="method", canonical_name="GraphRAG"),
            GraphNodeInput(entity_id="entity:method:eval-baseline", entity_type="method", canonical_name="Baseline"),
            GraphNodeInput(entity_id="entity:task:eval-reconstruction", entity_type="task", canonical_name="Reconstruction"),
        ]
    )
    neo4j_repository.upsert_relationships(
        [
            _rel("rel:eval:a-dataset", "paper:arxiv:eval-a", "entity:dataset:eval-shared", "evaluated_on", "chunk:eval:a-dataset"),
            _rel("rel:eval:a-method", "paper:arxiv:eval-a", "entity:method:eval-graphrag", "uses_method", "chunk:eval:a-method"),
            _rel("rel:eval:a-task", "paper:arxiv:eval-a", "entity:task:eval-reconstruction", "addresses", "chunk:eval:a-task"),
            _rel("rel:eval:b-dataset", "paper:arxiv:eval-b", "entity:dataset:eval-shared", "evaluated_on", "chunk:eval:b-dataset"),
            _rel("rel:eval:b-cites-a", "paper:arxiv:eval-b", "paper:arxiv:eval-a", "cites", "chunk:eval:b-cite"),
            _rel("rel:eval:c-method", "paper:arxiv:eval-c", "entity:method:eval-graphrag", "uses_method", "chunk:eval:c-method"),
            _rel("rel:eval:b-method", "paper:arxiv:eval-b", "entity:method:eval-baseline", "uses_method", "chunk:eval:b-method"),
        ]
    )

    settings = Settings(
        _env_file=None,
        EMBEDDING_PROVIDER="controlled_fake",
        EMBEDDING_MODEL="controlled-eval-4d-v1",
        EMBEDDING_NORMALIZE=False,
        VECTOR_SEARCH_DEFAULT_TOP_K=5,
        GRAPH_DEFAULT_LIMIT=20,
        HYBRID_DEFAULT_TOP_K=10,
        HYBRID_RRF_K=60,
    )
    service = HybridRetrievalService(
        vector_service=VectorSearchService(_EvalEmbeddingProvider(), vector_repo, default_top_k=5, max_top_k=10),
        graph_service=GraphRetrievalService(neo4j_repository, max_depth=3, default_limit=5, max_limit=10),
        provenance_bridge=EvidenceProvenanceBridge(
            vector_repo,
            max_supporting_chunks=5,
            expected_vector_generation_fingerprint="vector-current",
        ),
        fusion_service=EvidenceFusionService(rrf_k=60),
        default_top_k=10,
        max_top_k=20,
    )

    report = RetrievalEvaluationRunner(
        service,
        settings=settings,
        vector_candidate_count=5,
        graph_candidate_count=20,
        hybrid_top_k=10,
        metadata_overrides={"embedding_config_fingerprint": "controlled-test-fingerprint"},
    ).run_dataset(load_benchmark("evaluation/retrieval_benchmark.json"))
    json_path, markdown_path = write_reports(report, tmp_path)

    assert report.overall[RetrievalStrategy.VECTOR.value].applicable_cases == 10
    assert report.overall[RetrievalStrategy.GRAPH.value].applicable_cases == 8
    assert report.overall[RetrievalStrategy.GRAPH.value].not_applicable_cases == 2
    assert "struct-datasets-a" in report.graph_wins
    assert "struct-datasets-a" in report.hybrid_wins
    assert report.overall[RetrievalStrategy.HYBRID.value].hit_at_k[5] > 0
    assert Path(json_path).exists()
    assert Path(markdown_path).exists()


def _rel(
    relationship_id: str,
    source_entity_id: str,
    target_entity_id: str,
    relationship_type: str,
    source_chunk_id: str,
) -> GraphRelationshipInput:
    return GraphRelationshipInput(
        relationship_id=relationship_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relationship_type=relationship_type,
        confidence=0.9,
        extraction_version="extract-v1",
        source_chunk_id=source_chunk_id,
        supporting_chunk_ids=[source_chunk_id],
        provenance_type="chunk",
        paper_version_id=f"{source_entity_id}:v1",
        graph_index_generation_fingerprint="graph-current",
    )
