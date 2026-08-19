"""Deterministic canonical entity resolution (prompt #8-14).

No LLM call anywhere in this module (prompt #10) -- every decision is a
pure function of `(entity_type, canonical_name)` plus the explicit
`EntityAliasRegistry`. This is what turns one paper's extraction
candidates into entities that can safely be shared across papers in
Neo4j.

Key insight this module leans on (see `app.domain.ids.build_entity_id`):
`ScientificEntity.create()` already derives `entity_id` from a *Unicode-
normalized, whitespace-collapsed, case-folded* name -- Tier 2 ("exact
normalized match", prompt #9) is therefore already the behavior of the
existing id scheme, not something this resolver has to implement from
scratch. What this resolver actually *adds* is Tier 3 (explicit alias
retargeting) and the bookkeeping to merge/dedupe candidates that collapse
onto the same canonical entity once that retargeting is applied.
"""

from collections import defaultdict

from app.domain.enums import EntityType, RelationshipType
from app.domain.ids import build_relationship_id, normalize_identity_key
from app.domain.knowledge import ScientificEntity, ScientificRelationship
from app.ingestion.canonical_resolution.alias_registry import EntityAliasRegistry
from app.ingestion.canonical_resolution.models import CanonicalGraph, EntityResolution, ResolutionTier

# Bounded so repeatedly-seen evidence never bloats a shared canonical
# entity's metadata unreasonably across many papers (mirrors Prompt 8's
# `_MAX_STORED_EVIDENCE_QUOTES` for the same reason).
_MAX_STORED_ALIASES = 20


class CanonicalEntityResolver:
    """Resolves extraction-candidate entities to canonical identity, and
    remaps relationships onto that canonical identity."""

    def __init__(self, alias_registry: EntityAliasRegistry) -> None:
        self._aliases = alias_registry

    def resolve_entity(self, entity: ScientificEntity) -> EntityResolution:
        """Resolve one candidate entity (prompt #9's tier ladder)."""

        if entity.entity_type == EntityType.PAPER:
            # Tier 1: trusted identity is already `entity_id == paper_id`
            # (set by whoever built this entity -- `_build_paper_entity`
            # in Prompt 8, or the citation-target builder in
            # `GraphIndexingService`) -- never re-derived, never merged by
            # title (prompt #12/#18).
            return EntityResolution(
                candidate_entity_id=entity.entity_id,
                canonical_entity=entity,
                tier=ResolutionTier.TRUSTED_IDENTITY,
            )

        alias_target = self._aliases.resolve(entity.entity_type, entity.canonical_name)
        if alias_target is not None:
            # Tier 3: an explicit, reviewed alias entry retargets this
            # candidate's canonical name -- and therefore its entity_id,
            # since `ScientificEntity.create` derives it from the name.
            canonical = ScientificEntity.create(
                entity_type=entity.entity_type,
                canonical_name=alias_target,
                aliases=_merge_alias_lists(entity.aliases, [entity.canonical_name]),
                metadata=entity.metadata,
            )
            tier = ResolutionTier.EXPLICIT_ALIAS
        else:
            # Tier 1 (Author: normalized display name is already trusted
            # identity, prompt #17) / Tier 2 (Method/Dataset/Task: exact
            # normalized match) -- both already the effect of
            # `ScientificEntity.create`'s own id derivation; recomputing
            # it here is a deliberate no-op, not a missing step, kept for
            # a single obvious code path rather than a special case.
            canonical = ScientificEntity.create(
                entity_type=entity.entity_type,
                canonical_name=entity.canonical_name,
                aliases=entity.aliases,
                metadata=entity.metadata,
            )
            tier = (
                ResolutionTier.TRUSTED_IDENTITY
                if entity.entity_type == EntityType.AUTHOR
                else ResolutionTier.EXACT_NORMALIZED
            )

        return EntityResolution(
            candidate_entity_id=entity.entity_id, canonical_entity=canonical, tier=tier
        )

    def build_canonical_graph(
        self,
        *,
        paper_id: str,
        paper_version_id: str,
        entities: list[ScientificEntity],
        relationships: list[ScientificRelationship],
        canonicalization_version: str,
        canonicalization_config_fingerprint: str,
        graph_index_generation_fingerprint: str,
    ) -> CanonicalGraph:
        """Resolve every candidate entity, remap every relationship onto
        canonical identity, and deduplicate whatever collapses together.

        `entities` must already include a `ScientificEntity` for every
        `CITES` relationship target (real or placeholder) -- building
        those requires PostgreSQL access this module deliberately doesn't
        have, so `GraphIndexingService` constructs them before calling
        this method.
        """

        resolutions = [self.resolve_entity(entity) for entity in entities]
        remap = {r.candidate_entity_id: r.canonical_entity.entity_id for r in resolutions}

        canonical_entities = _dedupe_canonical_entities(resolutions)
        canonical_relationships = _remap_and_dedupe_relationships(relationships, remap)

        return CanonicalGraph(
            paper_id=paper_id,
            paper_version_id=paper_version_id,
            canonicalization_version=canonicalization_version,
            canonicalization_config_fingerprint=canonicalization_config_fingerprint,
            graph_index_generation_fingerprint=graph_index_generation_fingerprint,
            entities=canonical_entities,
            relationships=canonical_relationships,
            resolutions=resolutions,
        )


def _merge_alias_lists(*alias_lists: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for aliases in alias_lists:
        for alias in aliases:
            key = normalize_identity_key(alias)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(alias)
    return merged[:_MAX_STORED_ALIASES]


def _dedupe_canonical_entities(resolutions: list[EntityResolution]) -> list[ScientificEntity]:
    """Multiple candidates can resolve to the same canonical `entity_id`
    (e.g. two alias variants within one paper, or the same method
    mentioned via slightly different extraction candidates) -- merge them
    into one entity, conservatively unioning aliases and preserving the
    first-seen canonical name/metadata (prompt #40: never let a later,
    lower-quality candidate overwrite what's already there)."""

    by_id: dict[str, ScientificEntity] = {}
    for resolution in resolutions:
        entity = resolution.canonical_entity
        existing = by_id.get(entity.entity_id)
        if existing is None:
            by_id[entity.entity_id] = entity
            continue
        by_id[entity.entity_id] = existing.model_copy(
            update={"aliases": _merge_alias_lists(existing.aliases, entity.aliases)}
        )
    return list(by_id.values())


def _remap_and_dedupe_relationships(
    relationships: list[ScientificRelationship], remap: dict[str, str]
) -> list[ScientificRelationship]:
    """Rebuild every relationship's identity from its (possibly remapped)
    canonical endpoints, then merge any that now collide onto the same
    `(source, type, target)` -- the same shape as Prompt 8's
    `_dedupe_relationship_candidates`, one layer up."""

    groups: dict[tuple[str, RelationshipType, str], list[ScientificRelationship]] = defaultdict(list)
    for relationship in relationships:
        source_id = remap.get(relationship.source_entity_id, relationship.source_entity_id)
        target_id = remap.get(relationship.target_entity_id, relationship.target_entity_id)
        groups[(source_id, relationship.relationship_type, target_id)].append(relationship)

    resolved: list[ScientificRelationship] = []
    for (source_id, relationship_type, target_id), group in groups.items():
        supporting_chunk_ids: set[str] = set()
        evidence_quotes: list[str] = []
        for relationship in group:
            for chunk_id in relationship.metadata.get("supporting_chunk_ids", []):
                supporting_chunk_ids.add(chunk_id)
            if relationship.source_chunk_id:
                supporting_chunk_ids.add(relationship.source_chunk_id)
            for quote in relationship.metadata.get("evidence_quotes", []):
                if quote not in evidence_quotes:
                    evidence_quotes.append(quote)

        # First-seen non-null `source_chunk_id` -- never dropped, mirrors
        # Prompt 8's own provenance policy (CLAUDE.md #6).
        source_chunk_id = next((r.source_chunk_id for r in group if r.source_chunk_id), None)
        metadata = dict(group[0].metadata)
        if supporting_chunk_ids:
            metadata["supporting_chunk_ids"] = sorted(supporting_chunk_ids)
        if evidence_quotes:
            metadata["evidence_quotes"] = evidence_quotes

        resolved.append(
            ScientificRelationship(
                relationship_id=build_relationship_id(source_id, relationship_type, target_id),
                source_entity_id=source_id,
                target_entity_id=target_id,
                relationship_type=relationship_type,
                source_chunk_id=source_chunk_id,
                confidence=max(r.confidence for r in group),
                extraction_version=group[0].extraction_version,
                metadata=metadata,
            )
        )
    return resolved
