import asyncio
import json
from sqlalchemy import text
from db.connection import async_session, engine
from db.models import Base, KnowledgeBase
from core.ml_engine import ml_engine


async def migrate_data():
    print("Настройка базы данных и расширения pgvector...")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    ml_engine.load_models()

    with open("propositional_kb (1).json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    async with async_session() as db:
        try:
            print(f"Начинаем перенос {len(metadata)} записей в PostgreSQL...")
            for item in metadata:
                text_to_embed = f"{item['entity_name']}: {item['content']['natural_language']}"

                embedding = ml_engine.search_model.encode([text_to_embed])[0]

                entry = KnowledgeBase(
                    id=item['id'],
                    entity_id=item['entity_name'],
                    domain=item.get('domain'),
                    fact_type=item['type'],
                    content=item['content'],
                    relations=item['relations'],
                    embedding=embedding
                )
                db.add(entry)

            await db.commit()
            print("Миграция успешно завершена! Данные готовы к векторному поиску.")

        except Exception as e:
            print(f"Ошибка во время переноса данных: {e}")
            await db.rollback()


if __name__ == "__main__":
    asyncio.run(migrate_data())