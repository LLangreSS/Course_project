import asyncio
import json
import re

from sqlalchemy import text
from db.connection import async_session, engine
from db.models import Base, KnowledgeBase
from core.ml_engine import ml_engine

def tokenize_ru(txt: str):
    return re.findall(r'\w+', txt.lower())

async def migrate_data():
    print("Настройка базы данных и расширения pgvector...")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("CREATE INDEX idx_kb_search_text ON knowledge_base "
                                "USING gin(to_tsvector('russian', search_text_bm25))"))

    ml_engine.load_models()

    with open("propositional_kb (1).json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    async with async_session() as db:
        try:
            print(f"Начинаем перенос {len(metadata)} записей в PostgreSQL...")
            x = 0
            for item in metadata:
                print(x)
                print(item)
                x+=1
                embed_parts = [f"Термин: {item['entity_name']}. Определение {item['entity_name']}: "
                               f"{item['content']['natural_language']}"]

                if item['content']['explanation']:
                    embed_parts.append(f"Иными словами: {item['content']['explanation']}")
                if item['content']['examples']:
                    embed_parts.append(f"Примеры {item['entity_name']}: {'; '.join(item['content']['examples'])}")
                if item['relations']['conflicts_with']:
                    embed_parts.append(f"Это противоречит: {'; '.join(item['relations']['conflicts_with'])}")

                embed_text = " ".join(embed_parts)
                embedding = ml_engine.search_model.encode([embed_text], normalize_embeddings=True)[0]

                entry = KnowledgeBase(
                    id=item['id'],
                    entity_id=item['entity_name'],
                    domain=item.get('domain'),
                    fact_type=item['type'],
                    content=item['content'],
                    relations=item['relations'],
                    embedding=embedding,
                    search_text_bm25=embed_text
                )
                db.add(entry)

            await db.commit()
            print("Миграция успешно завершена! Данные готовы к векторному поиску.")

        except Exception as e:
            print(f"Ошибка во время переноса данных: {e}")
            await db.rollback()


if __name__ == "__main__":
    asyncio.run(migrate_data())