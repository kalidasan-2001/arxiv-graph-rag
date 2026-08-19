"""Declarative base shared by every PostgreSQL ORM model.

A fixed naming convention keeps constraint/index names stable and
predictable across Alembic autogenerate runs -- without it, SQLAlchemy lets
the database assign anonymous names, which then churn on every regeneration.
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for all PostgreSQL ORM models in this project."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
