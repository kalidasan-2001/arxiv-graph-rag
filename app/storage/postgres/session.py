"""SQLAlchemy engine and session factory.

No global mutable session: every caller gets a fresh `Session` from a
`SessionFactory` instance, used as a context manager so each unit of work's
transaction boundary (commit on success, rollback on error) is explicit at
the call site rather than hidden behind module state.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError


def build_engine(settings: Settings) -> Engine:
    """Create a SQLAlchemy engine from `settings.DATABASE_URL`.

    `pool_pre_ping` guards against stale connections after the database
    restarts or an idle connection is dropped by a proxy/firewall.
    """

    if not settings.DATABASE_URL:
        raise ConfigurationError(
            "DATABASE_URL is not configured; set it in the environment or .env"
        )
    return create_engine(settings.DATABASE_URL, pool_pre_ping=True, future=True)


class SessionFactory:
    """Callable that yields a transactional `Session` as a context manager.

    Usage::

        session_factory = SessionFactory()
        with session_factory() as session:
            repo = PaperRepository(session)
            repo.upsert_paper(paper)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.engine = build_engine(self._settings)
        self._sessionmaker = sessionmaker(
            bind=self.engine, expire_on_commit=False, future=True
        )

    @contextmanager
    def __call__(self) -> Iterator[Session]:
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


@lru_cache
def get_session_factory() -> SessionFactory:
    """Return a process-wide cached `SessionFactory` bound to `Settings.DATABASE_URL`.

    Cached so the connection pool is created once, not per request.
    """

    return SessionFactory()


def get_db_session() -> Iterator[Session]:
    """FastAPI dependency: `Depends(get_db_session)` yields a request-scoped session."""

    with get_session_factory()() as session:
        yield session
