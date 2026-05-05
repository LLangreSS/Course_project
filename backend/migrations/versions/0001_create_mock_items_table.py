"""Create mock_items table

Revision ID: 0001
Revises: 
Create Date: 2026-05-05 13:13:08.996336

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        """
        CALL paradedb.create_bm25_test_table(
          schema_name => 'public',
          table_name => 'knowledge_base'
        )
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF EXISTS public.knowledge_base")
