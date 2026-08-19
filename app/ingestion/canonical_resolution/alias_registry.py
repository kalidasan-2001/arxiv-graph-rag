"""The explicit, version-controlled alias registry (prompt #15/#16).

This is the *only* place a "these two surface forms mean the same entity"
decision is asserted. Canonical resolution never guesses via fuzzy string
similarity or acronym expansion (prompt #16: "KG" != "Knowledge Graph",
"RAG" != "Retrieval-Augmented Generation" as blanket equivalences) --
every alias here is a deliberate, reviewed, auditable entry, scoped to one
`EntityType` so e.g. an author-name alias can never accidentally apply to
a method name that happens to look similar.

V1 ships with an **empty** registry. That is a deliberate default, not an
oversight: CLAUDE.md #20 says "prefer unresolved/ambiguous status over an
incorrect merge," and the prompt's own MIMIC/MIMIC-IV and GraphRAG/
Microsoft GraphRAG examples are explicit that the default, unreviewed
behavior must be to keep them separate. Add entries here only after a
human has looked at the specific real names extracted and confirmed they
denote the same entity -- never speculatively.
"""

import hashlib
import json

from app.domain.enums import EntityType
from app.domain.ids import normalize_identity_key

REGISTRY_VERSION = "v1"

# Deliberate, reviewed alias declarations only -- see module docstring.
# Format: (entity_type, alias_surface_form) -> canonical_surface_form.
# Intentionally empty in V1.
_ALIASES: dict[tuple[EntityType, str], str] = {}


class EntityAliasRegistry:
    """A small, explicit alias-resolution mechanism (prompt #15), scoped by
    entity type. Tier 3 in `CanonicalEntityResolver`'s resolution ladder --
    the *only* tier that can make two differently-spelled candidates
    resolve to the same canonical entity."""

    def __init__(
        self,
        aliases: dict[tuple[EntityType, str], str] | None = None,
        *,
        version: str = REGISTRY_VERSION,
    ) -> None:
        # Keys are re-normalized here (not trusted as pre-normalized) so a
        # caller passing `(EntityType.METHOD, "Graph RAG")` and one passing
        # `(EntityType.METHOD, "graph rag")` are the same entry.
        source = aliases if aliases is not None else _ALIASES
        self._aliases: dict[tuple[EntityType, str], str] = {
            (entity_type, normalize_identity_key(alias)): canonical
            for (entity_type, alias), canonical in source.items()
        }
        self.version = version

    def resolve(self, entity_type: EntityType, name: str) -> str | None:
        """Return the canonical surface form for `name` under `entity_type`,
        or `None` if there is no explicit alias entry -- never a fuzzy
        guess (prompt #9's Tier 3: "only merge when the system has an
        explicit alias mapping")."""

        return self._aliases.get((entity_type, normalize_identity_key(name)))

    @property
    def checksum(self) -> str:
        """Deterministic fingerprint of the registry's actual content, so
        `canonicalization_config_fingerprint` changes whenever an alias is
        added/changed/removed -- not just when `version` is manually
        bumped (prompt #32's "do not rely only on a version string")."""

        canonical = sorted(
            (entity_type.value, alias, canonical)
            for (entity_type, alias), canonical in self._aliases.items()
        )
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_default_alias_registry() -> EntityAliasRegistry:
    """The application's real (V1: empty) alias registry."""

    return EntityAliasRegistry()
