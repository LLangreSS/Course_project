from typing import Optional, Dict, Any

from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB

from pgvector.sqlalchemy import Vector

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