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


def reciprocal_rank_fusion(vector_ids: list, fts_ids: list, k: int = 60):
    scores = {}
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(fts_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


@router.post('/', response_model=VerifyResponse)
async def verify_text(request: VerifyRequest, db: DB):
    atomic_claims = split_into_atomic_claims(request.text)
    print(atomic_claims)
    all_results = []
    has_contradiction = False

    for claim in atomic_claims:
        print(claim)
        query_vec_list = await run_in_threadpool(
            ml_engine.search_model.encode, [f"query: {claim}"], normalize_embeddings=True
        )
        query_vec = query_vec_list[0].tolist()

        sim_expr = (1 - KnowledgeBase.embedding.cosine_distance(query_vec)).label('similarity')
        vec_stmt = select(KnowledgeBase.id, sim_expr).order_by(sim_expr.desc()).limit(10)
        vec_res = await db.execute(vec_stmt)
        vec_rows = vec_res.all()

        # Фильтр по порогу как у напарника
        vector_candidates = [row.id for row in vec_rows if row.similarity >= (request.threshold - 0.15)]

        fts_stmt = (
            select(KnowledgeBase.id)
            .where(search.match_any(KnowledgeBase.search_text_bm25, claim))
            .order_by(desc(pdb.score(KnowledgeBase.id)))
            .limit(10)
        )
        fts_res = await db.execute(fts_stmt)
        fts_candidates = fts_res.scalars().all()

        fused_indices = reciprocal_rank_fusion(vector_candidates, list(fts_candidates))

        if not fused_indices:
            all_results.append(FactResult(fact=claim, verdict="Нет данных", max_sim=0.0))
            continue

        candidate_ids = fused_indices[:10]
        stmt = select(KnowledgeBase).where(KnowledgeBase.id.in_(candidate_ids))
        db_docs = await db.execute(stmt)
        id_to_doc = {doc.id: doc for doc in db_docs.scalars().all()}

        ordered_candidates = [id_to_doc[did] for did in candidate_ids if did in id_to_doc]
        pairs = [[claim, doc.search_text_bm25] for doc in ordered_candidates]

        rerank_scores = await run_in_threadpool(ml_engine.rerank_model.predict, pairs)
        reranked = sorted(zip(ordered_candidates, rerank_scores), key=lambda x: x[1], reverse=True)

        best_results = reranked[:3]

        final_contexts = [res[0].search_text_bm25 for res in best_results]
        probs_batch = await run_in_threadpool(run_nli_model, final_contexts, [claim] * len(final_contexts))

        current_fact_matches = []
        for i, probs in enumerate(probs_batch):
            doc, rerank_score = best_results[i]

            scores_dict = {
                ('Подтверждено' if 'entail' in ml_engine.id2label[j].lower() else
                 'Противоречие' if 'contradict' in ml_engine.id2label[j].lower() else
                 'Нейтрально'): p for j, p in enumerate(probs)
            }

            best_label = max(scores_dict, key=scores_dict.get)
            fact_verdict = "Недостаточно контекста" if (
                    best_label == 'Нейтрально' and scores_dict['Нейтрально'] > 0.6) else best_label

            current_fact_matches.append(MatchDetail(
                term=doc.entity_id,
                context=doc.content.get("natural_language", ""),
                similarity=float(rerank_score),
                scores=scores_dict,
                verdict=fact_verdict,
                subject_match=True
            ))

        top_match = current_fact_matches[0]
        if top_match.verdict == "Противоречие":
            has_contradiction = True

        all_results.append(FactResult(
            fact=claim,
            verdict=top_match.verdict,
            max_sim=top_match.similarity,
            best_match=top_match,
            all_matches=current_fact_matches
        ))

    missing_count = sum(1 for r in all_results if r.verdict == "Нет данных")
    if missing_count == len(all_results):
        final_status = "NO_DATA"
    elif missing_count > 0:
        final_status = "PARTIAL_SUCCESS"
    else:
        final_status = "SUCCESS"

    return VerifyResponse(
        status=final_status,
        has_contradiction=has_contradiction,
        results=all_results
    )