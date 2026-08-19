"""Deterministic ontology rules: the relationship compatibility matrix and
candidate validation (prompt #8/#9).

The LLM is never trusted to enforce ontology correctness on its own
(CLAUDE.md #7) -- every candidate it proposes passes through here before
becoming a trusted `ExtractedEntityCandidate`/`ExtractedRelationshipCandidate`.
A candidate that fails any rule is dropped with a `GraphExtractionWarning`,
never silently coerced into something the ontology doesn't actually support.
"""

from app.domain.enums import EntityType, RelationshipType
from app.domain.ids import normalize_whitespace
from app.ingestion.graph_extraction.models import (
    ExtractedEntityCandidate,
    ExtractedRelationshipCandidate,
    GraphExtractionWarning,
    GraphExtractionWarningCode,
    RawEntityCandidate,
    RawRelationshipCandidate,
    UsageClassification,
)

# Exactly the initial ontology (CLAUDE.md #5, prompt #2/#9) -- one valid
# (source_type, target_type) pair per relationship type in this V1 ontology.
# Rejects e.g. `Dataset -AUTHORED_BY-> Author` or `Method -CITES-> Paper`.
RELATIONSHIP_COMPATIBILITY: dict[RelationshipType, tuple[EntityType, EntityType]] = {
    RelationshipType.AUTHORED_BY: (EntityType.PAPER, EntityType.AUTHOR),
    RelationshipType.CITES: (EntityType.PAPER, EntityType.PAPER),
    RelationshipType.USES_METHOD: (EntityType.PAPER, EntityType.METHOD),
    RelationshipType.EVALUATED_ON: (EntityType.PAPER, EntityType.DATASET),
    RelationshipType.ADDRESSES: (EntityType.PAPER, EntityType.TASK),
}

# Only these relationship types carry a real "used vs merely mentioned"
# distinction (prompt #13/#14) -- ADDRESSES/AUTHORED_BY/CITES don't need it.
_USAGE_SENSITIVE_RELATIONSHIP_TYPES = frozenset(
    {RelationshipType.USES_METHOD, RelationshipType.EVALUATED_ON}
)


def _parse_entity_type(raw: str) -> EntityType | None:
    try:
        return EntityType(raw.strip().lower())
    except (ValueError, AttributeError):
        return None


def _parse_relationship_type(raw: str) -> RelationshipType | None:
    try:
        return RelationshipType(raw.strip().lower())
    except (ValueError, AttributeError):
        return None


def _parse_usage(raw: str | None) -> UsageClassification | None:
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in ("used_by_this_paper", "mentioned_only"):
        return normalized  # type: ignore[return-value]
    return None


def validate_entity_candidate(
    raw: RawEntityCandidate, *, source_chunk_id: str
) -> tuple[ExtractedEntityCandidate | None, GraphExtractionWarning | None]:
    """Validate one raw entity candidate against the ontology (prompt #8).

    Returns `(candidate, None)` on success, or `(None, warning)` if
    rejected -- never raises, since one bad candidate among many in a
    chunk's response must not fail the whole chunk.
    """

    entity_type = _parse_entity_type(raw.entity_type)
    if entity_type is None:
        return None, GraphExtractionWarning(
            code=GraphExtractionWarningCode.ENTITY_CANDIDATE_REJECTED,
            detail=f"unsupported entity_type {raw.entity_type!r} for {raw.name!r}",
            source_chunk_id=source_chunk_id,
        )

    if not normalize_whitespace(raw.name):
        return None, GraphExtractionWarning(
            code=GraphExtractionWarningCode.ENTITY_CANDIDATE_REJECTED,
            detail=f"blank name for entity_type {entity_type.value}",
            source_chunk_id=source_chunk_id,
        )

    if not 0.0 <= raw.confidence <= 1.0:
        return None, GraphExtractionWarning(
            code=GraphExtractionWarningCode.ENTITY_CANDIDATE_REJECTED,
            detail=f"confidence {raw.confidence} out of [0,1] for {raw.name!r}",
            source_chunk_id=source_chunk_id,
        )

    return (
        ExtractedEntityCandidate(
            entity_type=entity_type,
            name=raw.name,
            aliases=raw.aliases,
            evidence_quote=raw.evidence_quote,
            confidence=raw.confidence,
            source_chunk_id=source_chunk_id,
        ),
        None,
    )


def validate_relationship_candidate(
    raw: RawRelationshipCandidate, *, source_chunk_id: str, chunk_text: str
) -> tuple[ExtractedRelationshipCandidate | None, list[GraphExtractionWarning]]:
    """Validate one raw relationship candidate against the ontology and
    compatibility matrix (prompt #8/#9), and check its evidence quote
    against the source chunk (prompt #19).

    Returns `(candidate_or_none, warnings)`. A discarded evidence quote
    does not by itself reject the candidate -- only ontology/compatibility/
    confidence failures do.
    """

    warnings: list[GraphExtractionWarning] = []

    relationship_type = _parse_relationship_type(raw.relationship_type)
    if relationship_type is None:
        warnings.append(
            GraphExtractionWarning(
                code=GraphExtractionWarningCode.RELATIONSHIP_CANDIDATE_REJECTED,
                detail=f"unsupported relationship_type {raw.relationship_type!r}",
                source_chunk_id=source_chunk_id,
            )
        )
        return None, warnings

    source_type = _parse_entity_type(raw.source_type)
    target_type = _parse_entity_type(raw.target_type)
    if source_type is None or target_type is None:
        warnings.append(
            GraphExtractionWarning(
                code=GraphExtractionWarningCode.RELATIONSHIP_CANDIDATE_REJECTED,
                detail=(
                    f"unsupported source_type/target_type "
                    f"({raw.source_type!r}/{raw.target_type!r}) for {relationship_type.value}"
                ),
                source_chunk_id=source_chunk_id,
            )
        )
        return None, warnings

    expected = RELATIONSHIP_COMPATIBILITY.get(relationship_type)
    if expected is None or (source_type, target_type) != expected:
        warnings.append(
            GraphExtractionWarning(
                code=GraphExtractionWarningCode.RELATIONSHIP_CANDIDATE_REJECTED,
                detail=(
                    f"{source_type.value if source_type else '?'} -{relationship_type.value}-> "
                    f"{target_type.value if target_type else '?'} is not in the compatibility matrix"
                ),
                source_chunk_id=source_chunk_id,
            )
        )
        return None, warnings

    if not normalize_whitespace(raw.source_name) or not normalize_whitespace(raw.target_name):
        warnings.append(
            GraphExtractionWarning(
                code=GraphExtractionWarningCode.RELATIONSHIP_CANDIDATE_REJECTED,
                detail=f"blank source_name/target_name for {relationship_type.value}",
                source_chunk_id=source_chunk_id,
            )
        )
        return None, warnings

    if not 0.0 <= raw.confidence <= 1.0:
        warnings.append(
            GraphExtractionWarning(
                code=GraphExtractionWarningCode.RELATIONSHIP_CANDIDATE_REJECTED,
                detail=f"confidence {raw.confidence} out of [0,1] for {relationship_type.value}",
                source_chunk_id=source_chunk_id,
            )
        )
        return None, warnings

    usage = _parse_usage(raw.usage)
    if relationship_type in _USAGE_SENSITIVE_RELATIONSHIP_TYPES and usage != "used_by_this_paper":
        # Critical rule (prompt #13/#14): a method/dataset merely
        # mentioned (Related Work, prior work, etc.) never becomes a
        # trusted USES_METHOD/EVALUATED_ON edge -- only an explicit
        # "used_by_this_paper" classification does.
        warnings.append(
            GraphExtractionWarning(
                code=GraphExtractionWarningCode.RELATIONSHIP_CANDIDATE_REJECTED,
                detail=(
                    f"{relationship_type.value} for {raw.target_name!r} is not classified "
                    f"used_by_this_paper (usage={raw.usage!r}) -- mention only, not a trusted relation"
                ),
                source_chunk_id=source_chunk_id,
            )
        )
        return None, warnings

    evidence_quote = raw.evidence_quote
    if evidence_quote and not _quote_found_in_text(evidence_quote, chunk_text):
        warnings.append(
            GraphExtractionWarning(
                code=GraphExtractionWarningCode.EVIDENCE_QUOTE_DISCARDED,
                detail=f"evidence_quote not found verbatim in source chunk text: {evidence_quote!r}",
                source_chunk_id=source_chunk_id,
            )
        )
        evidence_quote = None  # discarded, but the relation itself still stands

    return (
        ExtractedRelationshipCandidate(
            relationship_type=relationship_type,
            source_name=raw.source_name,
            source_type=source_type,
            target_name=raw.target_name,
            target_type=target_type,
            usage=usage,
            evidence_quote=evidence_quote,
            confidence=raw.confidence,
            source_chunk_id=source_chunk_id,
        ),
        warnings,
    )


def _quote_found_in_text(quote: str, text: str) -> bool:
    """Prompt #19: never trust an LLM-authored quote blindly. Whitespace-
    normalized, case-insensitive substring check -- tolerant of the minor
    whitespace/line-break noise PDF text extraction introduces, but still
    requires the quote's actual words to appear in order in the source."""

    normalized_quote = normalize_whitespace(quote).lower()
    normalized_text = normalize_whitespace(text).lower()
    return bool(normalized_quote) and normalized_quote in normalized_text
