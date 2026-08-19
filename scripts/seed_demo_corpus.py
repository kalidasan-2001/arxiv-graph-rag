"""Seed a tiny local demo corpus for the frontend query workspace.

This is intentionally small and idempotent: it writes a handful of
GraphSteal paper/entity relationships to Neo4j and matching source chunks
to the configured Qdrant collection. It does not delete existing data.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.embeddings.provider import get_embedding_provider
from app.graph.client import get_neo4j_driver
from app.graph.models import GraphNodeInput, GraphRelationshipInput
from app.graph.neo4j_repository import Neo4jGraphRepository
from app.storage.qdrant.client import get_qdrant_client
from app.storage.qdrant.models import VectorPoint, VectorPointPayload, build_qdrant_point_id
from app.storage.qdrant.qdrant_repository import QdrantVectorRepository


DEMO_FINGERPRINT = "demo-graphsteal-v1"
PAPER_VERSION_GRAPHSTEAL = "paper-version:demo:graphsteal:v1"
PAPER_VERSION_GRAPHSHIELD = "paper-version:demo:graphshield:v1"


def main() -> None:
    settings = Settings()
    embedding_provider = get_embedding_provider()
    vector_repo = QdrantVectorRepository(get_qdrant_client(settings), settings.QDRANT_COLLECTION)
    graph_driver = get_neo4j_driver(settings)
    graph_repo = Neo4jGraphRepository(graph_driver, settings.NEO4J_DATABASE)

    chunks = _chunks(embedding_provider)
    vectors = embedding_provider.embed_documents([chunk.payload.text for chunk in chunks])
    points = [
        chunk.model_copy(update={"vector": vector})
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    vector_repo.ensure_collection(dimension=embedding_provider.dimension, distance="cosine")
    vector_repo.upsert_chunks(points)

    graph_repo.ensure_schema()
    graph_repo.upsert_entities(_nodes())
    graph_repo.upsert_relationships(_relationships())
    graph_driver.close()

    print(f"Seeded {len(points)} Qdrant chunks into {settings.QDRANT_COLLECTION!r}.")
    print("Seeded GraphSteal demo entities and relationships into Neo4j.")


def _nodes() -> list[GraphNodeInput]:
    return [
        GraphNodeInput(
            entity_id="paper:demo:graphsteal",
            entity_type="paper",
            canonical_name="GraphSteal",
            aliases=["GraphSteal: Graph Reconstruction Attacks"],
            properties={"source": "demo", "source_id": "graphsteal", "title": "GraphSteal"},
        ),
        GraphNodeInput(
            entity_id="paper:demo:graphshield",
            entity_type="paper",
            canonical_name="GraphShield",
            aliases=["GraphShield"],
            properties={"source": "demo", "source_id": "graphshield", "title": "GraphShield"},
        ),
        GraphNodeInput(
            entity_id="entity:method:graphsteal",
            entity_type="method",
            canonical_name="GraphSteal",
            aliases=["GraphSteal method"],
        ),
        GraphNodeInput(
            entity_id="entity:dataset:hotpotqa",
            entity_type="dataset",
            canonical_name="HotpotQA",
        ),
        GraphNodeInput(
            entity_id="entity:dataset:mimic-iv",
            entity_type="dataset",
            canonical_name="MIMIC-IV",
            aliases=["MIMIC IV"],
        ),
        GraphNodeInput(
            entity_id="entity:task:graph-reconstruction",
            entity_type="task",
            canonical_name="graph reconstruction attack",
        ),
    ]


def _relationships() -> list[GraphRelationshipInput]:
    return [
        _rel(
            "rel:demo:graphsteal-method",
            "paper:demo:graphsteal",
            "entity:method:graphsteal",
            "uses_method",
            "chunk:demo:graphsteal-method",
            PAPER_VERSION_GRAPHSTEAL,
        ),
        _rel(
            "rel:demo:graphsteal-hotpotqa",
            "paper:demo:graphsteal",
            "entity:dataset:hotpotqa",
            "evaluated_on",
            "chunk:demo:graphsteal-datasets",
            PAPER_VERSION_GRAPHSTEAL,
        ),
        _rel(
            "rel:demo:graphsteal-mimic",
            "paper:demo:graphsteal",
            "entity:dataset:mimic-iv",
            "evaluated_on",
            "chunk:demo:graphsteal-datasets",
            PAPER_VERSION_GRAPHSTEAL,
        ),
        _rel(
            "rel:demo:graphsteal-task",
            "paper:demo:graphsteal",
            "entity:task:graph-reconstruction",
            "addresses",
            "chunk:demo:graphsteal-method",
            PAPER_VERSION_GRAPHSTEAL,
        ),
        _rel(
            "rel:demo:graphshield-cites-graphsteal",
            "paper:demo:graphshield",
            "paper:demo:graphsteal",
            "cites",
            "chunk:demo:graphshield-citation",
            PAPER_VERSION_GRAPHSHIELD,
        ),
        _rel(
            "rel:demo:graphshield-method",
            "paper:demo:graphshield",
            "entity:method:graphsteal",
            "uses_method",
            "chunk:demo:graphshield-citation",
            PAPER_VERSION_GRAPHSHIELD,
        ),
        _rel(
            "rel:demo:graphshield-mimic",
            "paper:demo:graphshield",
            "entity:dataset:mimic-iv",
            "evaluated_on",
            "chunk:demo:graphshield-citation",
            PAPER_VERSION_GRAPHSHIELD,
        ),
    ]


def _rel(
    relationship_id: str,
    source_entity_id: str,
    target_entity_id: str,
    relationship_type: str,
    source_chunk_id: str,
    paper_version_id: str,
) -> GraphRelationshipInput:
    return GraphRelationshipInput(
        relationship_id=relationship_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relationship_type=relationship_type,
        confidence=0.95,
        extraction_version="demo",
        source_chunk_id=source_chunk_id,
        supporting_chunk_ids=[source_chunk_id],
        provenance_type="chunk",
        paper_version_id=paper_version_id,
        graph_index_generation_fingerprint=DEMO_FINGERPRINT,
    )


def _chunks(embedding_provider) -> list[VectorPoint]:
    payloads = [
        VectorPointPayload(
            chunk_id="chunk:demo:graphsteal-method",
            paper_id="paper:demo:graphsteal",
            paper_version_id=PAPER_VERSION_GRAPHSTEAL,
            section_id="section:demo:graphsteal:method",
            section_type="methodology",
            section_title="Methodology",
            chunk_index=0,
            page_start=1,
            page_end=2,
            source="demo",
            source_id="graphsteal",
            published_year=2026,
            categories=["cs.IR", "cs.AI"],
            chunking_version="demo",
            chunk_config_fingerprint=DEMO_FINGERPRINT,
            embedding_provider=embedding_provider.provider_name,
            embedding_model=embedding_provider.model_name,
            embedding_config_fingerprint=embedding_provider.config_fingerprint,
            vector_generation_fingerprint=DEMO_FINGERPRINT,
            text=(
                "GraphSteal proposes a graph reconstruction attack that combines entity linking, "
                "citation structure, and retrieval traces to infer missing relationships in a "
                "scientific knowledge graph."
            ),
        ),
        VectorPointPayload(
            chunk_id="chunk:demo:graphsteal-datasets",
            paper_id="paper:demo:graphsteal",
            paper_version_id=PAPER_VERSION_GRAPHSTEAL,
            section_id="section:demo:graphsteal:experiments",
            section_type="experiments",
            section_title="Experiments",
            chunk_index=1,
            page_start=3,
            page_end=4,
            source="demo",
            source_id="graphsteal",
            published_year=2026,
            categories=["cs.IR", "cs.AI"],
            chunking_version="demo",
            chunk_config_fingerprint=DEMO_FINGERPRINT,
            embedding_provider=embedding_provider.provider_name,
            embedding_model=embedding_provider.model_name,
            embedding_config_fingerprint=embedding_provider.config_fingerprint,
            vector_generation_fingerprint=DEMO_FINGERPRINT,
            text="GraphSteal evaluates the attack on HotpotQA and MIMIC-IV.",
        ),
        VectorPointPayload(
            chunk_id="chunk:demo:graphshield-citation",
            paper_id="paper:demo:graphshield",
            paper_version_id=PAPER_VERSION_GRAPHSHIELD,
            section_id="section:demo:graphshield:related-work",
            section_type="related_work",
            section_title="Related Work",
            chunk_index=0,
            page_start=1,
            page_end=2,
            source="demo",
            source_id="graphshield",
            published_year=2026,
            categories=["cs.IR", "cs.AI"],
            chunking_version="demo",
            chunk_config_fingerprint=DEMO_FINGERPRINT,
            embedding_provider=embedding_provider.provider_name,
            embedding_model=embedding_provider.model_name,
            embedding_config_fingerprint=embedding_provider.config_fingerprint,
            vector_generation_fingerprint=DEMO_FINGERPRINT,
            text=(
                "GraphShield cites GraphSteal, uses the GraphSteal method for comparison, "
                "and evaluates the defense on MIMIC-IV."
            ),
        ),
    ]
    return [
        VectorPoint(point_id=build_qdrant_point_id(payload.chunk_id), vector=[], payload=payload)
        for payload in payloads
    ]


if __name__ == "__main__":
    main()
