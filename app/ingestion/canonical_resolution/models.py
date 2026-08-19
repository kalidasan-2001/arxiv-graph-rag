"""DTOs for canonical entity resolution (prompt #11).

Reuses `ScientificEntity`/`ScientificRelationship` (Prompt 1) directly for
the resolved output -- there is no second, parallel
"CanonicalScientificEntity" model, per the prompt's explicit instruction
("reuse ScientificEntity where possible"). What's new here is only the
*resolution bookkeeping* around them: which tier resolved a candidate, and
the fully-resolved graph ready to hand to `GraphRepository`.
"""

from enum import Enum

from pydantic import BaseModel, Field

from app.domain.knowledge import ScientificEntity, ScientificRelationship


class ResolutionTier(str, Enum):
    """Which resolution rule produced a canonical entity (prompt #9)."""

    # Paper (its own `paper_id`) and Author (normalized display name) --
    # identity that is trusted without any lookup.
    TRUSTED_IDENTITY = "trusted_identity"
    # Same `(entity_type, normalize_identity_key(name))` as an
    # already-known entity -- literal-formatting variants only.
    EXACT_NORMALIZED = "exact_normalized"
    # An explicit `EntityAliasRegistry` entry retargeted this candidate to
    # a different canonical surface form.
    EXPLICIT_ALIAS = "explicit_alias"


class EntityResolution(BaseModel):
    """One candidate entity's resolution outcome -- kept for inspection/
    debugging (`scripts/resolve_entity.py`), not persisted."""

    candidate_entity_id: str
    canonical_entity: ScientificEntity
    tier: ResolutionTier


class CanonicalGraph(BaseModel):
    """The fully resolved, deduplicated graph for one paper version's
    generation -- ready to hand to `GraphRepository` (prompt #4). Every
    relationship's `source_entity_id`/`target_entity_id` already point at
    canonical (post-resolution) entity ids."""

    paper_id: str
    paper_version_id: str
    canonicalization_version: str
    canonicalization_config_fingerprint: str
    graph_index_generation_fingerprint: str
    entities: list[ScientificEntity] = Field(default_factory=list)
    relationships: list[ScientificRelationship] = Field(default_factory=list)
    resolutions: list[EntityResolution] = Field(default_factory=list)
