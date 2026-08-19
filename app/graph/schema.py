"""Idempotent Neo4j schema bootstrap.

Every uniqueness constraint/index this application needs lives in exactly
this one place (prompt #27) -- never scattered `CREATE CONSTRAINT` calls
across business/service code. `Neo4jGraphRepository.ensure_schema()`
delegates here.
"""

from neo4j import Driver

_ALL_LABELS = ("Paper", "Author", "Method", "Dataset", "Task")
# `canonical_name` is useful to index for every non-Paper label -- lookups
# during canonicalization/inspection filter on it. `Paper` is looked up by
# `entity_id`/`source_id`, never `canonical_name` (titles aren't a
# meaningful lookup key), so it's excluded here (prompt #26: "do not
# over-index every property").
_CANONICAL_NAME_INDEXED_LABELS = ("Author", "Method", "Dataset", "Task")


class GraphSchemaManager:
    """Owns constraint/index bootstrap for one Neo4j database."""

    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def ensure_schema(self) -> None:
        """Run every schema statement. `IF NOT EXISTS` makes this safe to
        call on every request (prompt #26/#48) -- never drops/recreates
        existing schema, so repeated calls are always a no-op after the
        first."""

        with self._driver.session(database=self._database) as session:
            for statement in self._statements():
                session.run(statement)

    def _statements(self) -> list[str]:
        statements: list[str] = []
        for label in _ALL_LABELS:
            statements.append(
                f"CREATE CONSTRAINT {label.lower()}_entity_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.entity_id IS UNIQUE"
            )
        for label in _CANONICAL_NAME_INDEXED_LABELS:
            statements.append(
                f"CREATE INDEX {label.lower()}_canonical_name IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.canonical_name)"
            )
        # `source_id` is the natural lookup key for citation-target
        # resolution (prompt #19/#26) -- a Paper node is looked up by
        # `entity_id` directly in every path this stage implements, but
        # this index is cheap and matches the prompt's explicit list.
        statements.append("CREATE INDEX paper_source_id IF NOT EXISTS FOR (n:Paper) ON (n.source_id)")
        return statements
