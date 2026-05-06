import asyncio
import json

from db.connection import async_session, engine
from db.models import Base, KnowledgeBase
from core.ml_engine import ml_engine


async def migrate_data():
    print("Настройка базы данных и расширения pgvector...")
    ml_engine.load_models()

    with open("parsed_data.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    async with async_session() as db:
        try:
            print(f"Начинаем перенос {len(metadata)} записей в PostgreSQL...")
            x = 0
            for item in metadata:
                print(x)
                x += 1
                print(item)
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

                rich_context = (f"Термин: {item['entity_name']} (Область: {item['domain']}). "
                                f"Определение: {item['content']['natural_language']} ")
                if item['content']['formal_logic']: rich_context += f"Формально: {item['content']['formal_logic']}. "
                if item['content']['explanation']: rich_context += f"Пояснение: {item['content']['explanation']}. "
                if item['relations']['conflicts_with']: rich_context += f"Логические анти-факты: {'; '.join(item['relations']['conflicts_with'])}."

                entry = KnowledgeBase(
                    id=item['id'],
                    entity_id=item['entity_name'],
                    domain=item.get('domain'),
                    fact_type=item['type'],
                    content=item['content'],
                    relations=item['relations'],
                    embedding=embedding,
                    search_text_bm25=embed_text,
                    rich_context=rich_context
                )
                db.add(entry)

            await db.commit()
            print("Миграция успешно завершена! Данные готовы к векторному поиску.")

        except Exception as e:
            print(f"Ошибка во время переноса данных: {e}")
            await db.rollback()


if __name__ == "__main__":
    asyncio.run(migrate_data())