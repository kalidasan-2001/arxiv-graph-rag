"""`GraphRepository` implementation backed by the official `neo4j` driver.

The only module in the codebase (besides `client.py`'s connection factory
and `schema.py`'s bootstrap) that imports `neo4j` directly -- everything
above this layer works with `app.graph.models` DTOs (CLAUDE.md #15). Every
query here is parameterized (prompt #30) -- no entity/relationship value
is ever interpolated into a Cypher string; only label/relationship-type
names are (they come from the fixed ontology dicts below, never from LLM
output or user input, per prompt #22).
"""

import logging

from neo4j import Driver
from neo4j.exceptions import DriverError, Neo4jError

from app.core.exceptions import GraphStoreUnavailableError
from app.graph.models import (
    GraphNodeInput,
    GraphNodeRecord,
    GraphPathRecord,
    GraphRelationshipInput,
    GraphRelationshipRecord,
)
from app.graph.schema import GraphSchemaManager

logger = logging.getLogger(__name__)

# Cypher labels/relationship types can't be parameterized -- these are the
# only values this repository ever interpolates into a query string, and
# both dicts are fixed, internal, and drawn from the closed ontology
# (`EntityType`/`RelationshipType`), never from extracted text (prompt #22).
_LABEL_BY_ENTITY_TYPE = {
    "paper": "Paper",
    "author": "Author",
    "method": "Method",
    "dataset": "Dataset",
    "task": "Task",
}
_CYPHER_TYPE_BY_RELATIONSHIP_TYPE = {
    "authored_by": "AUTHORED_BY",
    "cites": "CITES",
    "uses_method": "USES_METHOD",
    "evaluated_on": "EVALUATED_ON",
    "addresses": "ADDRESSES",
}
_ENTITY_TYPE_BY_LABEL = {label: entity_type for entity_type, label in _LABEL_BY_ENTITY_TYPE.items()}

# Node/relationship properties that are surfaced through dedicated
# `GraphNodeRecord`/`GraphRelationshipRecord` fields rather than the
# generic `properties`/free-form dict -- excluded when building the latter
# so it never duplicates them.
_NODE_RESERVED_KEYS = {"entity_id", "entity_type", "canonical_name", "aliases"}
_REL_RESERVED_KEYS = {
    "relationship_id",
    "confidence",
    "extraction_version",
    "source_chunk_id",
    "supporting_chunk_ids",
    "provenance_type",
    "paper_version_id",
    "graph_index_generation_fingerprint",
}


class Neo4jGraphRepository:
    """Talks to one configured Neo4j database."""

    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database
        self._schema = GraphSchemaManager(driver, database)

    def ensure_schema(self) -> None:
        try:
            self._schema.ensure_schema()
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"could not reach Neo4j: {exc}") from exc

    def upsert_entities(self, nodes: list[GraphNodeInput]) -> None:
        if not nodes:
            return
        by_label: dict[str, list[dict]] = {}
        for node in nodes:
            label = _LABEL_BY_ENTITY_TYPE[node.entity_type]
            by_label.setdefault(label, []).append(
                {
                    "entity_id": node.entity_id,
                    "canonical_name": node.canonical_name,
                    "entity_type": node.entity_type,
                    "aliases": node.aliases,
                    "properties": node.properties,
                }
            )

        query_template = (
            "UNWIND $nodes AS node "
            "MERGE (n:{label} {{entity_id: node.entity_id}}) "
            "ON CREATE SET n.canonical_name = node.canonical_name, "
            "              n.entity_type = node.entity_type, "
            "              n.aliases = node.aliases, "
            "              n += node.properties "
            "ON MATCH SET  n.aliases = REDUCE(acc = coalesce(n.aliases, []), a IN node.aliases | "
            "                CASE WHEN a IN acc THEN acc ELSE acc + a END), "
            "              n += node.properties"
        )
        try:
            with self._driver.session(database=self._database) as session:
                for label, batch in by_label.items():
                    session.run(query_template.format(label=label), nodes=batch)
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j entity upsert failed: {exc}") from exc

    def upsert_relationships(self, relationships: list[GraphRelationshipInput]) -> None:
        if not relationships:
            return
        by_type: dict[str, list[dict]] = {}
        for rel in relationships:
            cypher_type = _CYPHER_TYPE_BY_RELATIONSHIP_TYPE[rel.relationship_type]
            by_type.setdefault(cypher_type, []).append(
                {
                    "relationship_id": rel.relationship_id,
                    "source_entity_id": rel.source_entity_id,
                    "target_entity_id": rel.target_entity_id,
                    "confidence": rel.confidence,
                    "extraction_version": rel.extraction_version,
                    "source_chunk_id": rel.source_chunk_id,
                    "supporting_chunk_ids": rel.supporting_chunk_ids,
                    "provenance_type": rel.provenance_type,
                    "paper_version_id": rel.paper_version_id,
                    "graph_index_generation_fingerprint": rel.graph_index_generation_fingerprint,
                }
            )

        # Endpoint nodes are `MATCH`ed, never `MERGE`d/created here --
        # `GraphIndexingService` always upserts entities first, so a
        # missing endpoint means a real bug, not something to paper over.
        # A missing endpoint simply produces no row for that `UNWIND` item
        # (no relationship written); the service's post-upsert
        # relationship-id verification (prompt #50) is what turns that
        # into a loud failure rather than a silent gap.
        query_template = (
            "UNWIND $rels AS rel "
            "MATCH (s {{entity_id: rel.source_entity_id}}) "
            "MATCH (t {{entity_id: rel.target_entity_id}}) "
            "MERGE (s)-[r:{cypher_type} {{relationship_id: rel.relationship_id}}]->(t) "
            "SET r.confidence = rel.confidence, "
            "    r.extraction_version = rel.extraction_version, "
            "    r.source_chunk_id = rel.source_chunk_id, "
            "    r.supporting_chunk_ids = rel.supporting_chunk_ids, "
            "    r.provenance_type = rel.provenance_type, "
            "    r.paper_version_id = rel.paper_version_id, "
            "    r.graph_index_generation_fingerprint = rel.graph_index_generation_fingerprint"
        )
        try:
            with self._driver.session(database=self._database) as session:
                for cypher_type, batch in by_type.items():
                    session.run(query_template.format(cypher_type=cypher_type), rels=batch)
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j relationship upsert failed: {exc}") from exc

    def get_relationship_ids_for_generation(
        self, paper_version_id: str, *, generation_fingerprint: str
    ) -> set[str]:
        query = (
            "MATCH ()-[r]->() "
            "WHERE r.paper_version_id = $paper_version_id "
            "  AND r.graph_index_generation_fingerprint = $generation_fingerprint "
            "RETURN r.relationship_id AS relationship_id"
        )
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(
                    query, paper_version_id=paper_version_id, generation_fingerprint=generation_fingerprint
                )
                return {record["relationship_id"] for record in result}
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j read failed: {exc}") from exc

    def delete_generation(self, paper_version_id: str, *, exclude_generation_fingerprint: str) -> int:
        query = (
            "MATCH ()-[r]->() "
            "WHERE r.paper_version_id = $paper_version_id "
            "  AND r.graph_index_generation_fingerprint <> $exclude_generation_fingerprint "
            "DELETE r "
            "RETURN count(r) AS deleted"
        )
        try:
            with self._driver.session(database=self._database) as session:
                record = session.run(
                    query,
                    paper_version_id=paper_version_id,
                    exclude_generation_fingerprint=exclude_generation_fingerprint,
                ).single()
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j stale-generation cleanup failed: {exc}") from exc
        return record["deleted"] if record else 0

    def get_entity(self, entity_id: str) -> GraphNodeRecord | None:
        query = "MATCH (n {entity_id: $entity_id}) RETURN n"
        try:
            with self._driver.session(database=self._database) as session:
                record = session.run(query, entity_id=entity_id).single()
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j read failed: {exc}") from exc
        return _node_to_record(record["n"]) if record else None

    def get_paper_graph(
        self, paper_id: str
    ) -> tuple[GraphNodeRecord | None, list[GraphNodeRecord], list[GraphRelationshipRecord]]:
        try:
            with self._driver.session(database=self._database) as session:
                paper_record = session.run(
                    "MATCH (p:Paper {entity_id: $paper_id}) RETURN p", paper_id=paper_id
                ).single()
                if paper_record is None:
                    return None, [], []
                paper = _node_to_record(paper_record["p"])

                edges = session.run(
                    "MATCH (p:Paper {entity_id: $paper_id})-[r]->(t) RETURN r, t", paper_id=paper_id
                )
                nodes: dict[str, GraphNodeRecord] = {}
                relationships: list[GraphRelationshipRecord] = []
                for record in edges:
                    target = _node_to_record(record["t"])
                    nodes[target.entity_id] = target
                    relationships.append(
                        _relationship_to_record(
                            record["r"], source_entity_id=paper_id, target_entity_id=target.entity_id
                        )
                    )
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j read failed: {exc}") from exc
        return paper, list(nodes.values()), relationships

    def get_entity_context(
        self, entity_id: str
    ) -> tuple[GraphNodeRecord, list[GraphNodeRecord], list[GraphRelationshipRecord]] | None:
        try:
            with self._driver.session(database=self._database) as session:
                entity_record = session.run(
                    "MATCH (e {entity_id: $entity_id}) RETURN e", entity_id=entity_id
                ).single()
                if entity_record is None:
                    return None
                entity = _node_to_record(entity_record["e"])

                nodes: dict[str, GraphNodeRecord] = {}
                relationships: list[GraphRelationshipRecord] = []

                outbound = session.run(
                    "MATCH (e {entity_id: $entity_id})-[r]->(t) RETURN r, t", entity_id=entity_id
                )
                for record in outbound:
                    target = _node_to_record(record["t"])
                    nodes[target.entity_id] = target
                    relationships.append(
                        _relationship_to_record(
                            record["r"], source_entity_id=entity_id, target_entity_id=target.entity_id
                        )
                    )

                inbound = session.run(
                    "MATCH (s)-[r]->(e {entity_id: $entity_id}) RETURN r, s", entity_id=entity_id
                )
                for record in inbound:
                    source = _node_to_record(record["s"])
                    nodes[source.entity_id] = source
                    relationships.append(
                        _relationship_to_record(
                            record["r"], source_entity_id=source.entity_id, target_entity_id=entity_id
                        )
                    )
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j read failed: {exc}") from exc
        return entity, list(nodes.values()), relationships

    def find_entities_by_canonical_name(
        self, canonical_name: str, *, entity_type: str | None = None, limit: int = 20
    ) -> list[GraphNodeRecord]:
        params = {"canonical_name": canonical_name, "limit": limit}
        if entity_type is None:
            query = (
                "MATCH (n) "
                "WHERE n.entity_id IS NOT NULL "
                "  AND toLower(n.canonical_name) = toLower($canonical_name) "
                "RETURN n ORDER BY n.entity_type, n.entity_id LIMIT $limit"
            )
        else:
            label = _LABEL_BY_ENTITY_TYPE[entity_type]
            query = (
                f"MATCH (n:{label}) "
                "WHERE toLower(n.canonical_name) = toLower($canonical_name) "
                "RETURN n ORDER BY n.entity_id LIMIT $limit"
            )
        try:
            with self._driver.session(database=self._database) as session:
                return [_node_to_record(record["n"]) for record in session.run(query, **params)]
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j entity lookup failed: {exc}") from exc

    def get_direct_paths(
        self,
        start_entity_id: str,
        *,
        relationship_type: str,
        direction: str,
        end_entity_type: str | None = None,
        limit: int = 20,
    ) -> list[GraphPathRecord]:
        cypher_type = _CYPHER_TYPE_BY_RELATIONSHIP_TYPE[relationship_type]
        end_label = f":{_LABEL_BY_ENTITY_TYPE[end_entity_type]}" if end_entity_type else ""
        if direction == "outgoing":
            pattern = f"(start)-[r:{cypher_type}]->(end{end_label})"
            source_expr = "start.entity_id"
            target_expr = "end.entity_id"
        elif direction == "incoming":
            pattern = f"(end{end_label})-[r:{cypher_type}]->(start)"
            source_expr = "end.entity_id"
            target_expr = "start.entity_id"
        else:
            raise ValueError("direction must be 'outgoing' or 'incoming'")

        query = (
            f"MATCH {pattern} "
            "WHERE start.entity_id = $start_entity_id "
            f"RETURN start, end, r, {source_expr} AS source_id, {target_expr} AS target_id "
            "ORDER BY r.confidence DESC, end.entity_id ASC LIMIT $limit"
        )
        try:
            with self._driver.session(database=self._database) as session:
                records = session.run(query, start_entity_id=start_entity_id, limit=limit)
                return [
                    GraphPathRecord(
                        nodes=[_node_to_record(record["start"]), _node_to_record(record["end"])],
                        relationships=[
                            _relationship_to_record(
                                record["r"],
                                source_entity_id=record["source_id"],
                                target_entity_id=record["target_id"],
                            )
                        ],
                    )
                    for record in records
                ]
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j direct graph query failed: {exc}") from exc

    def get_shared_entity_paths(
        self,
        paper_id: str,
        *,
        relationship_type: str,
        shared_entity_type: str,
        limit: int = 20,
    ) -> list[GraphPathRecord]:
        cypher_type = _CYPHER_TYPE_BY_RELATIONSHIP_TYPE[relationship_type]
        shared_label = _LABEL_BY_ENTITY_TYPE[shared_entity_type]
        query = (
            f"MATCH (paper:Paper {{entity_id: $paper_id}})-[r1:{cypher_type}]->"
            f"(shared:{shared_label})<-[r2:{cypher_type}]-(other:Paper) "
            "WHERE other.entity_id <> $paper_id "
            "RETURN paper, shared, other, r1, r2 "
            "ORDER BY shared.entity_id ASC, other.entity_id ASC LIMIT $limit"
        )
        try:
            with self._driver.session(database=self._database) as session:
                return [_shared_path_to_record(record) for record in session.run(query, paper_id=paper_id, limit=limit)]
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j shared-entity query failed: {exc}") from exc

    def get_citing_paper_entity_paths(
        self,
        paper_id: str,
        *,
        relationship_type: str,
        end_entity_type: str,
        limit: int = 20,
    ) -> list[GraphPathRecord]:
        cypher_type = _CYPHER_TYPE_BY_RELATIONSHIP_TYPE[relationship_type]
        end_label = _LABEL_BY_ENTITY_TYPE[end_entity_type]
        query = (
            f"MATCH (citing:Paper)-[cite:CITES]->(paper:Paper {{entity_id: $paper_id}}), "
            f"      (citing)-[r:{cypher_type}]->(end:{end_label}) "
            "RETURN paper, citing, end, cite, r "
            "ORDER BY end.entity_id ASC, citing.entity_id ASC LIMIT $limit"
        )
        try:
            with self._driver.session(database=self._database) as session:
                return [
                    GraphPathRecord(
                        nodes=[
                            _node_to_record(record["paper"]),
                            _node_to_record(record["citing"]),
                            _node_to_record(record["end"]),
                        ],
                        relationships=[
                            _relationship_to_record(
                                record["cite"],
                                source_entity_id=record["citing"]["entity_id"],
                                target_entity_id=record["paper"]["entity_id"],
                            ),
                            _relationship_to_record(
                                record["r"],
                                source_entity_id=record["citing"]["entity_id"],
                                target_entity_id=record["end"]["entity_id"],
                            ),
                        ],
                    )
                    for record in session.run(query, paper_id=paper_id, limit=limit)
                ]
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j citing-paper query failed: {exc}") from exc

    def get_entity_paper_entity_paths(
        self,
        entity_id: str,
        *,
        incoming_relationship_type: str,
        outgoing_relationship_type: str,
        end_entity_type: str,
        limit: int = 20,
    ) -> list[GraphPathRecord]:
        incoming_type = _CYPHER_TYPE_BY_RELATIONSHIP_TYPE[incoming_relationship_type]
        outgoing_type = _CYPHER_TYPE_BY_RELATIONSHIP_TYPE[outgoing_relationship_type]
        end_label = _LABEL_BY_ENTITY_TYPE[end_entity_type]
        query = (
            f"MATCH (start {{entity_id: $entity_id}})<-[r1:{incoming_type}]-(paper:Paper)-"
            f"[r2:{outgoing_type}]->(end:{end_label}) "
            "RETURN start, paper, end, r1, r2 "
            "ORDER BY end.entity_id ASC, paper.entity_id ASC LIMIT $limit"
        )
        try:
            with self._driver.session(database=self._database) as session:
                return [
                    GraphPathRecord(
                        nodes=[
                            _node_to_record(record["start"]),
                            _node_to_record(record["paper"]),
                            _node_to_record(record["end"]),
                        ],
                        relationships=[
                            _relationship_to_record(
                                record["r1"],
                                source_entity_id=record["paper"]["entity_id"],
                                target_entity_id=record["start"]["entity_id"],
                            ),
                            _relationship_to_record(
                                record["r2"],
                                source_entity_id=record["paper"]["entity_id"],
                                target_entity_id=record["end"]["entity_id"],
                            ),
                        ],
                    )
                    for record in session.run(query, entity_id=entity_id, limit=limit)
                ]
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j entity-paper-entity query failed: {exc}") from exc

    def get_citation_neighborhood_paths(
        self, paper_id: str, *, depth: int = 1, limit: int = 20
    ) -> list[GraphPathRecord]:
        if depth == 1:
            queries = [
                (
                    "MATCH (paper:Paper {entity_id: $paper_id})-[r:CITES]->(other:Paper) "
                    "RETURN paper, other, r, paper.entity_id AS source_id, other.entity_id AS target_id "
                    "ORDER BY other.entity_id ASC LIMIT $limit",
                    "paper",
                    "other",
                ),
                (
                    "MATCH (other:Paper)-[r:CITES]->(paper:Paper {entity_id: $paper_id}) "
                    "RETURN paper, other, r, other.entity_id AS source_id, paper.entity_id AS target_id "
                    "ORDER BY other.entity_id ASC LIMIT $limit",
                    "paper",
                    "other",
                ),
            ]
        elif depth == 2:
            queries = [
                (
                    "MATCH (paper:Paper {entity_id: $paper_id})-[r1:CITES]->(mid:Paper)-[r2:CITES]->(other:Paper) "
                    "RETURN paper, mid, other, r1, r2 "
                    "ORDER BY mid.entity_id ASC, other.entity_id ASC LIMIT $limit",
                    "outgoing",
                ),
                (
                    "MATCH (other:Paper)-[r2:CITES]->(mid:Paper)-[r1:CITES]->(paper:Paper {entity_id: $paper_id}) "
                    "RETURN paper, mid, other, r1, r2 "
                    "ORDER BY mid.entity_id ASC, other.entity_id ASC LIMIT $limit",
                    "incoming",
                ),
            ]
        else:
            raise ValueError("citation neighborhood depth must be 1 or 2")

        paths: list[GraphPathRecord] = []
        try:
            with self._driver.session(database=self._database) as session:
                for query, *shape in queries:
                    for record in session.run(query, paper_id=paper_id, limit=limit):
                        if depth == 1:
                            paths.append(
                                GraphPathRecord(
                                    nodes=[_node_to_record(record["paper"]), _node_to_record(record["other"])],
                                    relationships=[
                                        _relationship_to_record(
                                            record["r"],
                                            source_entity_id=record["source_id"],
                                            target_entity_id=record["target_id"],
                                        )
                                    ],
                                )
                            )
                        else:
                            direction = shape[0]
                            if direction == "outgoing":
                                r1_source = record["paper"]["entity_id"]
                                r1_target = record["mid"]["entity_id"]
                                r2_source = record["mid"]["entity_id"]
                                r2_target = record["other"]["entity_id"]
                            else:
                                r1_source = record["mid"]["entity_id"]
                                r1_target = record["paper"]["entity_id"]
                                r2_source = record["other"]["entity_id"]
                                r2_target = record["mid"]["entity_id"]
                            paths.append(
                                GraphPathRecord(
                                    nodes=[
                                        _node_to_record(record["paper"]),
                                        _node_to_record(record["mid"]),
                                        _node_to_record(record["other"]),
                                    ],
                                    relationships=[
                                        _relationship_to_record(
                                            record["r1"],
                                            source_entity_id=r1_source,
                                            target_entity_id=r1_target,
                                        ),
                                        _relationship_to_record(
                                            record["r2"],
                                            source_entity_id=r2_source,
                                            target_entity_id=r2_target,
                                        ),
                                    ],
                                )
                            )
        except (Neo4jError, DriverError) as exc:
            raise GraphStoreUnavailableError(f"Neo4j citation-neighborhood query failed: {exc}") from exc
        return paths[:limit]


def _node_to_record(node) -> GraphNodeRecord:
    properties = dict(node)
    return GraphNodeRecord(
        entity_id=properties["entity_id"],
        entity_type=properties["entity_type"],
        canonical_name=properties["canonical_name"],
        aliases=list(properties.get("aliases") or []),
        properties={k: v for k, v in properties.items() if k not in _NODE_RESERVED_KEYS},
    )


def _relationship_to_record(rel, *, source_entity_id: str, target_entity_id: str) -> GraphRelationshipRecord:
    # `source_entity_id`/`target_entity_id` are passed in by the caller
    # (already known from the query's own MATCH pattern) rather than read
    # via `rel.start_node`/`end_node` -- those are only fully hydrated
    # with properties when the endpoint node itself is also part of the
    # query's RETURN clause, which isn't always true here (e.g. `p` in
    # `get_paper_graph`'s edge query).
    properties = dict(rel)
    return GraphRelationshipRecord(
        relationship_id=properties["relationship_id"],
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relationship_type=rel.type.lower(),
        confidence=properties["confidence"],
        extraction_version=properties.get("extraction_version"),
        source_chunk_id=properties.get("source_chunk_id"),
        supporting_chunk_ids=list(properties.get("supporting_chunk_ids") or []),
        provenance_type=properties["provenance_type"],
        paper_version_id=properties["paper_version_id"],
        graph_index_generation_fingerprint=properties["graph_index_generation_fingerprint"],
    )


def _shared_path_to_record(record) -> GraphPathRecord:
    paper = _node_to_record(record["paper"])
    shared = _node_to_record(record["shared"])
    other = _node_to_record(record["other"])
    return GraphPathRecord(
        nodes=[paper, shared, other],
        relationships=[
            _relationship_to_record(
                record["r1"], source_entity_id=paper.entity_id, target_entity_id=shared.entity_id
            ),
            _relationship_to_record(
                record["r2"], source_entity_id=other.entity_id, target_entity_id=shared.entity_id
            ),
        ],
    )


def get_graph_repository() -> "Neo4jGraphRepository":
    """FastAPI dependency (mirrors `get_vector_repository`/`get_llm_provider`
    -- no arguments, reads `Settings` itself) so tests can override it via
    `app.dependency_overrides` instead of a real Neo4j instance."""

    from app.core.config import get_settings
    from app.graph.client import get_neo4j_driver

    settings = get_settings()
    return Neo4jGraphRepository(get_neo4j_driver(settings), settings.NEO4J_DATABASE)
