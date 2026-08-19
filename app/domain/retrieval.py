"""Retrieval planning domain model.

Defines the shape of a retrieval request only -- no retrieval logic, no
Qdrant/Neo4j calls. See CLAUDE.md #21: vector, graph, and hybrid retrieval
must remain independently testable strategies that a future retrieval
subsystem implements against this plan.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.domain.enums import RetrievalStrategy
from app.domain.ids import ensure_json_safe, normalize_whitespace

# Domain-level sanity bound, not the final operational limit. The real
# MAX_GRAPH_DEPTH ceiling (CLAUDE.md #60) will come from application
# configuration once the retrieval subsystem exists; this only rejects
# nonsensical plans at construction time.
_MAX_GRAPH_DEPTH = 5


class RetrievalPlan(BaseModel):
    """A storage-independent description of how a query should be retrieved."""

    strategy: RetrievalStrategy
    query: str
    entity_ids: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    graph_operation: str | None = None
    graph_request: dict[str, Any] | None = None
    vector_top_k: int | None = None
    graph_limit: int | None = None
    final_top_k: int | None = None
    resolved_entities: list[dict[str, Any]] = Field(default_factory=list)
    requires_multiple_graph_operations: bool = False
    requested_graph_operations: list[str] = Field(default_factory=list)
    planner_metadata: dict[str, Any] = Field(default_factory=dict)
    graph_depth: int = 1
    top_k: int = 10

    @field_validator("query")
    @classmethod
    def _query_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized

    @field_validator("graph_depth")
    @classmethod
    def _graph_depth_bounds(cls, value: int) -> int:
        if not 1 <= value <= _MAX_GRAPH_DEPTH:
            raise ValueError(f"graph_depth must be between 1 and {_MAX_GRAPH_DEPTH}")
        return value

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be > 0")
        return value

    @field_validator("filters")
    @classmethod
    def _filters_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)

    @field_validator("graph_request", "planner_metadata")
    @classmethod
    def _optional_dict_json_safe(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return ensure_json_safe(value)

    @field_validator("resolved_entities")
    @classmethod
    def _resolved_entities_json_safe(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [ensure_json_safe(item) for item in value]
