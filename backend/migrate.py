import asyncio
import json
import pickle
import re
import os
from sqlalchemy import text
from rank_bm25 import BM25Okapi
from db.connection import async_session, engine
from db.models import Base, KnowledgeBase
from core.ml_engine import ml_engine

# Вспомогательная функция токенизации для BM25
def tokenize_ru(text):
    return re.findall(r'\w+', text.lower())

async def migrate_data():
    print("--- Настройка базы данных и расширения pgvector ---")
    async with engine.begin() as conn:
        # Создаем расширение для векторов, если его еще нет[cite: 1]
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Создаем таблицы на основе моделей SQLAlchemy[cite: 1]
        await conn.run_sync(Base.metadata.create_all)

    # Загружаем ML-модели для генерации эмбеддингов[cite: 1]
    ml_engine.load_models()

    # Формируем абсолютный путь к JSON, чтобы избежать FileNotFoundError[cite: 1]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "propositional_kb (1).json")

    if not os.path.exists(json_path):
        print(f"Ошибка: Файл {json_path} не найден!")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tokenized_corpus = [] # Сюда соберем данные для BM25[cite: 8, 9]

    async with async_session() as db:
        try:
            print(f"--- Загрузка {len(metadata)} объектов в PostgreSQL ---")
            for item in metadata:
                # 1. Подготовка текста для эмбеддинга (Entity Boosting)[cite: 9]
                # Усиливаем вес названия сущности для более точного векторного поиска
                entity = item.get('entity_name', '')
                natural_lang = item['content'].get('natural_language', '')
                explanation = item['content'].get('explanation', '')
                
                text_to_embed = f"Термин: {entity}. Определение {entity}: {natural_lang}"
                if explanation:
                    text_to_embed += f" Пояснение: {explanation}"

                # Генерируем вектор через SentenceTransformer[cite: 1, 9]
                embedding = ml_engine.search_model.encode([text_to_embed], normalize_embeddings=True)[0]

                # 2. Подготовка данных для BM25 индекса[cite: 8, 9]
                # Собираем все ключевые слова для лексического поиска
                full_text_for_bm25 = f"{entity} {natural_lang} {item.get('domain', '')} {explanation}"
                tokenized_corpus.append(tokenize_ru(full_text_for_bm25))

                # 3. Сохранение в базу данных[cite: 1]
                entry = KnowledgeBase(
                    id=item['id'],
                    entity_id=entity,
                    domain=item.get('domain'),
                    fact_type=item['type'],
                    content=item['content'],
                    relations=item['relations'],
                    embedding=embedding
                )
                db.add(entry)

            await db.commit()
            print("Данные успешно загружены в PostgreSQL.")

            # 4. Создание и сохранение индекса BM25[cite: 8, 9]
            print("--- Создание и сохранение индекса BM25 ---")
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_path = os.path.join(current_dir, "bm25_index.pkl")
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25, f)
            print(f"Индекс BM25 сохранен в {bm25_path}")

        except Exception as e:
            print(f"Ошибка во время миграции: {e}")
            await db.rollback()

if __name__ == "__main__":
    asyncio.run(migrate_data())