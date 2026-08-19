"""Shared paper/paper-version resolution.

Both `PdfAcquisitionService` (Prompt 4) and `PaperParsingService`
(Prompt 5) need identical "resolve `paper_id`, and either use the given
`paper_version_id` or fall back to the latest discovered version" logic.
This is a real business policy (not just a utility), so it lives in
exactly one place -- letting it drift between the two stages would be a
correctness bug, not just duplication.
"""

from app.core.exceptions import PaperNotFoundError, PaperVersionNotFoundError
from app.domain.papers import Paper, PaperVersion
from app.storage.postgres.repositories.papers import PaperRepository


def resolve_paper(paper_repository: PaperRepository, paper_id: str) -> Paper:
    paper = paper_repository.get_by_paper_id(paper_id)
    if paper is None:
        raise PaperNotFoundError(f"paper {paper_id} not found; discover it via search first")
    return paper


def resolve_version(
    paper_repository: PaperRepository, paper: Paper, paper_version_id: str | None
) -> PaperVersion:
    """Documented policy: when no version is specified, use the latest
    discovered version for the logical paper -- never silently invent one."""

    if paper_version_id is not None:
        version = paper_repository.get_paper_version(paper_version_id)
        if version is None or version.paper_id != paper.paper_id:
            raise PaperVersionNotFoundError(
                f"paper_version {paper_version_id} not found for paper {paper.paper_id}"
            )
        return version

    versions = paper_repository.list_versions(paper.paper_id)
    if not versions:
        raise PaperVersionNotFoundError(f"paper {paper.paper_id} has no discovered versions")
    return max(versions, key=lambda v: _version_ordinal(v.version))


def _version_ordinal(version: str) -> int:
    digits = "".join(ch for ch in version if ch.isdigit())
    return int(digits) if digits else -1
