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
        query_vec_list = await run_in_threadpool(
            ml_engine.search_model.encode, [f"query: {claim}"], normalize_embeddings=True
        )
        query_vec = query_vec_list[0].tolist()

        sim_expr = (1 - KnowledgeBase.embedding.cosine_distance(query_vec)).label('similarity')
        vec_stmt = select(KnowledgeBase.id, sim_expr).order_by(sim_expr.desc()).limit(10)
        vec_res = await db.execute(vec_stmt)
        vec_rows = vec_res.all()
        vector_candidates = [row.id for row in vec_rows if row.similarity >= (request.threshold - 0.15)]

        fts_stmt = (
            select(KnowledgeBase.id)
            .where(search.match_any(KnowledgeBase.search_text_bm25, claim))
            .order_by(desc(pdb.score(KnowledgeBase.id)))
            .limit(10)
        )
        fts_res = await db.execute(fts_stmt)
        fts_candidates = fts_res.scalars().all()

        rrf_scores_map = get_normalized_rrf_scores(vector_candidates, list(fts_candidates))
        fused_indices = sorted(rrf_scores_map.keys(), key=lambda x: rrf_scores_map[x], reverse=True)

        if not fused_indices:
            all_results.append(FactResult(fact=claim, verdict="Нет данных", max_sim=0.0))
            continue

        candidate_ids = fused_indices[:10]
        stmt = select(KnowledgeBase).where(KnowledgeBase.id.in_(candidate_ids))
        db_docs = await db.execute(stmt)
        id_to_doc = {doc.id: doc for doc in db_docs.scalars().all()}

        ordered_candidates = [id_to_doc[did] for did in candidate_ids if did in id_to_doc]

        pairs = [[claim, doc.rich_context] for doc in ordered_candidates]
        rerank_scores = await run_in_threadpool(ml_engine.rerank_model.predict, pairs)

        hybrid_results = []
        for doc, r_score in zip(ordered_candidates, rerank_scores):
            rrf_weight = rrf_scores_map.get(doc.id, 0)
            final_h_score = (r_score * 0.7) + (rrf_weight * 0.3)
            hybrid_results.append((doc, final_h_score))

        reranked = sorted(hybrid_results, key=lambda x: x[1], reverse=True)

        best_results = reranked[:3]
        best_doc, best_hybrid_score = best_results[0]

        final_contexts = [res[0].rich_context for res in best_results]
        probs_batch = await run_in_threadpool(run_nli_model, final_contexts, [claim] * len(final_contexts))
        best_probs = probs_batch[0]
        best_doc, best_hybrid_score = best_results[0]

        p_entail = 0.0
        p_contra = 0.0
        p_neutral = 0.0

        for j, p in enumerate(best_probs):
            label = ml_engine.id2label[j].lower()
            if 'entail' in label:
                p_entail = p
            elif 'contradict' in label:
                p_contra = p
            else:
                p_neutral = p

        scores_dict = {
            'Подтверждено': p_entail,
            'Противоречие': p_contra,
            'Нейтрально': p_neutral
        }

        if p_neutral >= 0.65:
            fact_verdict = "Недостаточно контекста"
        elif (p_neutral + p_contra) > p_entail:
            fact_verdict = "Противоречие"
        else:
            fact_verdict = "Подтверждено"

        top_match = MatchDetail(
            term=best_doc.entity_id,
            context=best_doc.content.get("natural_language", ""),
            similarity=float(best_hybrid_score),
            scores=scores_dict
        )

        all_results.append(FactResult(
            fact=claim,
            verdict=fact_verdict,
            best_match=top_match
        ))

    has_any_contradiction = any(r.verdict == "Противоречие" for r in all_results)
    has_any_no_data = any(r.verdict in ("Недостаточно контекста", "Нет данных") for r in all_results)

    if has_any_contradiction:
        final_status = "CONTRADICTION"
    elif has_any_no_data:
        final_status = "NO_DATA"
    else:
        final_status = "SUCCESS"

    return VerifyResponse(
        status=final_status,
        results=all_results
    )