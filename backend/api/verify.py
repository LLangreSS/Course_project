from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import torch
import numpy as np
import re

from db.connection import get_session
from db.models import KnowledgeBase
from core.ml_engine import ml_engine
from api.models import VerifyRequest, VerifyResponse, ClaimResult

router = APIRouter()


def tokenize_ru(text: str):
    return re.findall(r'\w+', text.lower())


def reciprocal_rank_fusion(faiss_ids, bm25_ids, k=60):
    """Точная копия твоей функции слияния из app3.py"""
    rrf_scores = {}
    for rank, doc_id in enumerate(faiss_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)


@router.post("/", response_model=VerifyResponse)
async def verify_text(request: VerifyRequest, db: AsyncSession = Depends(get_session)):
    if not ml_engine.search_model:
        ml_engine.load_models()

    # 1. Разбиение на атомарные утверждения (твоя регулярка)
    split_pattern = r'(?:,\s*а\s+)|(?:,\s*но\s+)|(?:,\s*и\s+)'
    claims = [p.strip() for p in re.split(split_pattern, request.text.strip().rstrip('.')) if
              len(p.strip().split()) >= 3]
    if not claims: claims = [request.text]

    # Загружаем все данные из БД один раз для BM25 маппинга
    # (В реальном проде лучше хранить маппинг в памяти, но для курсовой так надежнее)
    all_kb_res = await db.execute(select(KnowledgeBase))
    all_kb_items = {item.id: item for item in all_kb_res.scalars().all()}
    all_ids_ordered = list(all_kb_items.keys())

    final_results = []
    threshold = 0.65  # Твой дефолт из слайдера

    for claim in claims:
        # --- ШАГ 2: ГИБРИДНЫЙ ПОИСК ---

        # А) FAISS (через pgvector)
        query_vec = ml_engine.search_model.encode([f"query: {claim}"], normalize_embeddings=True)[0].tolist()
        # Ищем кандидатов для FAISS (valid_faiss в твоем коде)
        # Берем чуть больше (20), чтобы отфильтровать по порогу
        vector_res = await db.execute(
            select(KnowledgeBase, KnowledgeBase.embedding.cosine_distance(query_vec).label("dist"))
            .order_by("dist").limit(10)
        )
        # В твоем коде: valid_faiss = [idx for i, idx in enumerate(faiss_idx[0]) if faiss_dist[0][i] >= (threshold - 0.15)]
        # Важно: cosine_distance = 1 - similarity. Поэтому порог инвертируем.
        faiss_candidates = []
        for row in vector_res:
            sim = 1 - row.dist
            if sim >= (threshold - 0.15):
                faiss_candidates.append(row.KnowledgeBase.id)

        # Б) BM25
        tokenized_query = tokenize_ru(claim)
        bm25_scores = ml_engine.bm25.get_scores(tokenized_query)
        bm25_idx = np.argsort(bm25_scores)[::-1][:10].tolist()
        # В твоем коде: valid_bm25 = [idx for idx in bm25_idx if bm25_scores[idx] > 0]
        bm25_candidates = [all_ids_ordered[idx] for idx in bm25_idx if bm25_scores[idx] > 0]

        # В) RRF Fusion
        fused_ids = reciprocal_rank_fusion(faiss_candidates, bm25_candidates)

        if not fused_ids:
            final_results.append(
                ClaimResult(claim=claim, verdict="Нет данных", confidence=0.0, source_text="Данные не найдены"))
            continue

        # --- ШАГ 3: RERANKING (Cross-Encoder) ---
        candidate_items = [all_kb_items[tid] for tid in fused_ids[:10]]
        candidate_contexts = []
        for cand in candidate_items:
            # Формируем rich_context точно как в твоем парсере
            ctx = f"Термин: {cand.entity_id}. Определение: {cand.content.get('natural_language')} "
            if cand.content.get('explanation'): ctx += f"Пояснение: {cand.content.get('explanation')} "
            candidate_contexts.append(ctx)

        pairs = [[claim, ctx] for ctx in candidate_contexts]
        rerank_scores = ml_engine.rerank_model.predict(pairs)

        # Сортируем и берем топ-3 для NLI
        reranked_data = sorted(zip(candidate_items, rerank_scores, candidate_contexts), key=lambda x: x[1],
                               reverse=True)
        best_results = reranked_data[:3]

        # --- ШАГ 4: NLI ВЕРИФИКАЦИЯ ---
        final_contexts = [res[2] for res in best_results]
        inputs = ml_engine.tokenizer(final_contexts, [claim] * len(final_contexts), truncation=True, padding=True,
                                     return_tensors="pt").to(ml_engine.device)

        with torch.no_grad():
            outputs = ml_engine.nli_model(**inputs)
            probs_batch = torch.softmax(outputs.logits, dim=1).tolist()

        # Берем самый лучший результат из топ-3 (как в твоем UI)
        best_nli_probs = probs_batch[0]
        best_meta = best_results[0][0]

        # Маппинг лейблов (твоя логика из app3.py)
        id2label = ml_engine.id2label
        scores_dict = {}
        for j, p in enumerate(best_nli_probs):
            label = id2label[j].lower()
            key = 'Подтверждено' if 'entail' in label else 'Противоречие' if 'contradict' in label else 'Нейтрально'
            scores_dict[key] = p

        best_label = max(scores_dict, key=scores_dict.get)
        # Твоя логика "Недостаточно контекста"
        final_verdict = "Недостаточно контекста" if (
                    best_label == 'Нейтрально' and scores_dict['Нейтрально'] > 0.6) else best_label

        final_results.append(ClaimResult(
            claim=claim,
            verdict=final_verdict.lower(),
            confidence=float(scores_dict[best_label]),
            source_text=best_results[0][2],
            entity_name=best_meta.entity_id
        ))

    return VerifyResponse(results=final_results)