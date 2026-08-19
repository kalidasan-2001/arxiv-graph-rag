#!/usr/bin/env python
"""Manual development script: show how one name would resolve under
canonical entity resolution (prompt #77) -- useful for debugging Prompt 9
without needing a real extraction/Neo4j round trip.

Usage:
    python scripts/resolve_entity.py dataset "MIMIC-IV"
    python scripts/resolve_entity.py method " GraphRAG "
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.enums import EntityType  # noqa: E402
from app.domain.ids import normalize_identity_key  # noqa: E402
from app.domain.knowledge import ScientificEntity  # noqa: E402
from app.ingestion.canonical_resolution.alias_registry import get_default_alias_registry  # noqa: E402
from app.ingestion.canonical_resolution.resolver import CanonicalEntityResolver  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        print('usage: python scripts/resolve_entity.py <entity_type> "<name>"', file=sys.stderr)
        raise SystemExit(1)

    try:
        entity_type = EntityType(sys.argv[1].strip().lower())
    except ValueError:
        valid = ", ".join(t.value for t in EntityType)
        print(f"invalid entity_type {sys.argv[1]!r}; must be one of: {valid}", file=sys.stderr)
        raise SystemExit(1)
    name = sys.argv[2]

    candidate = ScientificEntity.create(entity_type=entity_type, canonical_name=name)
    resolver = CanonicalEntityResolver(get_default_alias_registry())
    resolution = resolver.resolve_entity(candidate)

    print(f"input:      {name!r}")
    print(f"normalized: {normalize_identity_key(name)!r}")
    print(f"canonical:  {resolution.canonical_entity.canonical_name!r}")
    print(f"entity_id:  {resolution.canonical_entity.entity_id}")
    print(f"aliases:    {resolution.canonical_entity.aliases}")
    print(f"tier:       {resolution.tier.value}")


if __name__ == "__main__":
    main()
