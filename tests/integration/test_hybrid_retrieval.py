"""Real Qdrant + Neo4j integration for explicit hybrid retrieval."""

from app.graph.models import GraphNodeInput, GraphRelationshipInput
from app.graph.neo4j_repository import Neo4jGraphRepository
from app.retrieval.evidence import EvidenceProvenanceBridge
from app.retrieval.graph_search import GraphRetrievalService, GraphSearchOperation
from app.retrieval.hybrid import EvidenceFusionService, HybridRetrievalService
from app.retrieval.vector_search import VectorSearchService
from app.storage.qdrant.models import VectorPoint, VectorPointPayload, build_qdrant_point_id
from app.storage.qdrant.qdrant_repository import QdrantVectorRepository
from app.domain.enums import EvidenceType, RetrievalStrategy


class _FakeEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


def _payload(chunk_id: str, text: str, vector_generation: str = "vector-current") -> VectorPointPayload:
    return VectorPointPayload(
        chunk_id=chunk_id,
        paper_id="paper:arxiv:hybrid-a",
        paper_version_id="paper-version:arxiv:hybrid-a:v1",
        section_id="section:hybrid-method",
        section_type="methodology",
        section_title="Method",
        chunk_index=0,
        page_start=4,
        page_end=5,
        source="arxiv",
        source_id="hybrid-a",
        published_year=2026,
        categories=["cs.CL"],
        chunking_version="chunk-v1",
        chunk_config_fingerprint="chunk-fp",
        embedding_provider="fake",
        embedding_model="fake-model",
        embedding_config_fingerprint="embed-fp",
        vector_generation_fingerprint=vector_generation,
        text=text,
    )


class TestHybridRetrievalIntegration:
    def test_hybrid_retrieval_links_vector_text_to_graph_evidence(
        self, qdrant_client, qdrant_collection_name, neo4j_repository: Neo4jGraphRepository
    ) -> None:
        vector_repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        vector_repo.ensure_collection(dimension=4, distance="cosine")
        payload = _payload("chunk:hybrid-support", "Hybrid source chunk mentions the dataset.")
        vector_repo.upsert_chunks(
            [
                VectorPoint(
                    point_id=build_qdrant_point_id(payload.chunk_id),
                    vector=[1.0, 0.0, 0.0, 0.0],
                    payload=payload,
                ),
                VectorPoint(
                    point_id=build_qdrant_point_id("chunk:hybrid-other"),
                    vector=[0.0, 1.0, 0.0, 0.0],
                    payload=_payload("chunk:hybrid-other", "Other chunk."),
                ),
            ]
        )

        neo4j_repository.ensure_schema()
        neo4j_repository.upsert_entities(
            [
                GraphNodeInput(
                    entity_id="paper:arxiv:hybrid-a",
                    entity_type="paper",
                    canonical_name="Hybrid Paper A",
                    properties={"title": "Hybrid Paper A"},
                ),
                GraphNodeInput(
                    entity_id="entity:dataset:hybrid",
                    entity_type="dataset",
                    canonical_name="Hybrid Dataset",
                ),
            ]
        )
        neo4j_repository.upsert_relationships(
            [
                GraphRelationshipInput(
                    relationship_id="rel:hybrid-dataset",
                    source_entity_id="paper:arxiv:hybrid-a",
                    target_entity_id="entity:dataset:hybrid",
                    relationship_type="evaluated_on",
                    confidence=0.91,
                    extraction_version="extract-v1",
                    source_chunk_id="chunk:hybrid-support",
                    supporting_chunk_ids=["chunk:hybrid-support"],
                    provenance_type="chunk",
                    paper_version_id="paper-version:arxiv:hybrid-a:v1",
                    graph_index_generation_fingerprint="graph-current",
                )
            ]
        )

        result = HybridRetrievalService(
            vector_service=VectorSearchService(
                _FakeEmbeddingProvider(), vector_repo, default_top_k=2, max_top_k=5
            ),
            graph_service=GraphRetrievalService(
                neo4j_repository, max_depth=3, default_limit=5, max_limit=10
            ),
            provenance_bridge=EvidenceProvenanceBridge(
                vector_repo,
                max_supporting_chunks=5,
                expected_vector_generation_fingerprint="vector-current",
            ),
            fusion_service=EvidenceFusionService(rrf_k=60),
            default_top_k=10,
            max_top_k=50,
        ).retrieve(
            query="hybrid dataset",
            strategy=RetrievalStrategy.HYBRID,
            vector_top_k=2,
            graph_operation=GraphSearchOperation.PAPER_DATASETS,
            entity_id="paper:arxiv:hybrid-a",
            top_k=10,
        )

        evidence_types = [item.evidence.evidence_type for item in result.evidence]
        assert EvidenceType.TEXT in evidence_types
        assert EvidenceType.GRAPH_RELATIONSHIP in evidence_types
        assert result.diagnostics["vector_candidates"] == 2
        assert result.diagnostics["graph_candidates"] == 1
        assert result.diagnostics["cross_store_links"] == 1
        graph_item = next(
            item for item in result.evidence if item.evidence.evidence_type == EvidenceType.GRAPH_RELATIONSHIP
        )
        assert graph_item.cross_store_supported is True
        assert graph_item.evidence.supporting_text_evidence_ids
        assert result.evidence_pool.items[0].pool_id == "E1"
