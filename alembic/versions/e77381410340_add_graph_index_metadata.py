"""add graph index metadata

Revision ID: e77381410340
Revises: d002a4409dd0
Create Date: 2026-08-18 17:42:13.615495

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e77381410340'
down_revision: Union[str, Sequence[str], None] = 'd002a4409dd0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('paper_versions', sa.Column('graph_indexed_at', postgresql.TIMESTAMP(timezone=True), nullable=True))
    op.add_column('paper_versions', sa.Column('canonical_entity_count', sa.Integer(), nullable=True))
    op.add_column('paper_versions', sa.Column('graph_relationship_count', sa.Integer(), nullable=True))
    op.add_column('paper_versions', sa.Column('canonicalization_config_fingerprint', sa.String(length=128), nullable=True))
    op.add_column('paper_versions', sa.Column('graph_index_generation_fingerprint', sa.String(length=128), nullable=True))
    op.add_column('paper_versions', sa.Column('neo4j_database', sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('paper_versions', 'neo4j_database')
    op.drop_column('paper_versions', 'graph_index_generation_fingerprint')
    op.drop_column('paper_versions', 'canonicalization_config_fingerprint')
    op.drop_column('paper_versions', 'graph_relationship_count')
    op.drop_column('paper_versions', 'canonical_entity_count')
    op.drop_column('paper_versions', 'graph_indexed_at')
