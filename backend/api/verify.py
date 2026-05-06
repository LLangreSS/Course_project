import re
import torch

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select, desc
from paradedb.sqlalchemy import search, pdb

from core.ml_engine import ml_engine
from db.models import KnowledgeBase
from schemas import VerifyResponse, VerifyRequest, FactResult, MatchDetail
from db.connection import DB

router = APIRouter(prefix='/verify', tags=['Verification'])


def split_into_atomic_claims(txt: str):
    """Разделяет сложный текст на атомарные утверждения по союзам."""
    txt = txt.strip().rstrip('.')
    split_pattern = r'(?:,?\s*\bа\b\s+)|(?:,?\s*\bно\b\s+)|(?:,?\s*\bи\b\s+)|(?:,?\s*\bгде\b\s+)'
    parts = re.split(split_pattern, txt)
    return [p.strip() for p in parts if len(p.strip().split()) >= 3] or [txt]


def run_nli_model(contexts: list[str], claims: list[str]):
    inputs = ml_engine.tokenizer(
        contexts, claims, truncation=True, padding=True,
        max_length=512, return_tensors="pt"
    )
    inputs = {k: v.to(ml_engine.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = ml_engine.nli_model(**inputs)
        return torch.softmax(outputs.logits, dim=1).tolist()


def get_normalized_rrf_scores(vector_ids: list, fts_ids: list, k: int = 60):
    """
    Считает RRF и нормализует баллы в диапазон [0, 1].
    Это необходимо для корректного смешивания со скорами реранкера.
    """
    scores = {}
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(fts_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    if not scores:
        return {}

    # Нормализация (Min-Max)
    max_s = max(scores.values())
    min_s = min(scores.values())

    if max_s == min_s:
        return {idx: 1.0 for idx in scores}

    return {idx: (s - min_s) / (max_s - min_s) for idx, s in scores.items()}


@router.post('/', response_model=VerifyResponse)
async def verify_text(request: VerifyRequest, db: DB):
    atomic_claims = split_into_atomic_claims(request.text)
    all_results = []
    has_contradiction = False

    for claim in atomic_claims:
        # 1. Поиск кандидатов (Векторный)
        query_vec_list = await run_in_threadpool(
            ml_engine.search_model.encode, [f"query: {claim}"], normalize_embeddings=True
        )
        query_vec = query_vec_list[0].tolist()

        sim_expr = (1 - KnowledgeBase.embedding.cosine_distance(query_vec)).label('similarity')
        vec_stmt = select(KnowledgeBase.id, sim_expr).order_by(sim_expr.desc()).limit(10)
        vec_res = await db.execute(vec_stmt)
        vec_rows = vec_res.all()
        vector_candidates = [row.id for row in vec_rows if row.similarity >= (request.threshold - 0.15)]

        # 2. Поиск кандидатов (Полнотекстовый BM25)
        fts_stmt = (
            select(KnowledgeBase.id)
            .where(search.match_any(KnowledgeBase.search_text_bm25, claim))
            .order_by(desc(pdb.score(KnowledgeBase.id)))
            .limit(10)
        )
        fts_res = await db.execute(fts_stmt)
        fts_candidates = fts_res.scalars().all()

        # 3. Слияние и получение нормализованных весов RRF
        rrf_scores_map = get_normalized_rrf_scores(vector_candidates, list(fts_candidates))
        fused_indices = sorted(rrf_scores_map.keys(), key=lambda x: rrf_scores_map[x], reverse=True)

        if not fused_indices:
            all_results.append(FactResult(fact=claim, verdict="Нет данных", max_sim=0.0))
            continue

        # 4. Загрузка документов
        candidate_ids = fused_indices[:10]
        stmt = select(KnowledgeBase).where(KnowledgeBase.id.in_(candidate_ids))
        db_docs = await db.execute(stmt)
        id_to_doc = {doc.id: doc for doc in db_docs.scalars().all()}

        # Сохраняем порядок, который был после RRF
        ordered_candidates = [id_to_doc[did] for did in candidate_ids if did in id_to_doc]

        # 5. Реранкинг (Cross-Encoder)
        pairs = [[claim, doc.rich_context] for doc in ordered_candidates]
        rerank_scores = await run_in_threadpool(ml_engine.rerank_model.predict, pairs)

        # 6. Гибридное взвешивание (Hybrid Scoring)
        # FinalScore = (Reranker * 0.7) + (RRF * 0.3)
        hybrid_results = []
        for doc, r_score in zip(ordered_candidates, rerank_scores):
            rrf_weight = rrf_scores_map.get(doc.id, 0)
            final_h_score = (r_score * 0.7) + (rrf_weight * 0.3)
            hybrid_results.append((doc, final_h_score))

        # Сортируем по итоговому гибридному баллу
        reranked = sorted(hybrid_results, key=lambda x: x[1], reverse=True)

        # Логгирование для отладки
        print(f"\n[ CLAIM: {claim} ]")
        for res, h_score in reranked:
            print(f"Hybrid: {h_score:.4f} | Entity: {res.entity_id}")

        best_results = reranked[:3]
        best_doc, best_hybrid_score = best_results[0]

        # 7. NLI Проверка (на основе лучшего контекста)
        final_contexts = [res[0].rich_context for res in best_results]
        probs_batch = await run_in_threadpool(run_nli_model, final_contexts, [claim] * len(final_contexts))
        best_probs = probs_batch[0]

        scores_dict = {
            ('Подтверждено' if 'entail' in ml_engine.id2label[j].lower() else
             'Противоречие' if 'contradict' in ml_engine.id2label[j].lower() else
             'Нейтрально'): p for j, p in enumerate(best_probs)
        }

        best_label = max(scores_dict, key=scores_dict.get)
        fact_verdict = "Недостаточно контекста" if (
                best_label == 'Нейтрально' and scores_dict['Нейтрально'] > 0.6) else best_label

        top_match = MatchDetail(
            term=best_doc.entity_id,
            context=best_doc.content.get("natural_language", ""),
            similarity=float(best_hybrid_score),
            scores=scores_dict,
            verdict=fact_verdict,
            subject_match=True
        )

        if top_match.verdict == "Противоречие":
            has_contradiction = True

        all_results.append(FactResult(
            fact=claim,
            verdict=top_match.verdict,
            max_sim=top_match.similarity,
            best_match=top_match,
            all_matches=[top_match]
        ))

    # Формирование итогового статуса
    missing_count = sum(1 for r in all_results if r.verdict == "Нет данных")
    if has_contradiction:
        final_status = "CONTRADICTION"
    elif missing_count == len(all_results):
        final_status = "NO_DATA"
    elif all(r.verdict == "Подтверждено" for r in all_results):
        final_status = "SUCCESS"
    else:
        final_status = "PARTIAL_SUCCESS"

    return VerifyResponse(
        status=final_status,
        has_contradiction=has_contradiction,
        results=all_results
    )