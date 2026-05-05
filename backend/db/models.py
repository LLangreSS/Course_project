from typing import Optional, Dict, Any

from sqlalchemy import String, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB

from pgvector.sqlalchemy import Vector
from paradedb.sqlalchemy import indexing


class Base(DeclarativeBase):
    pass


class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'

    id: Mapped[str] = mapped_column(String(50), primary_key=True)

    entity_id: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[Optional[str]] = mapped_column(String(100))
    fact_type: Mapped[str] = mapped_column(String(50))
    content: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    relations: Mapped[Dict[str, Any]] = mapped_column(JSONB)

    embedding: Mapped[Any] = mapped_column(Vector(384))
    search_text_bm25: Mapped[str] = mapped_column(String)

    rich_context: Mapped[str] = mapped_column(String)


Index(
    "search_idx",
    indexing.BM25Field(KnowledgeBase.id),
    indexing.BM25Field(KnowledgeBase.search_text_bm25),
    postgresql_using="bm25",
    postgresql_with={"key_field": "id"},
)




