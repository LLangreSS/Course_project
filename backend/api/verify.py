import re
import torch

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select

from core.ml_engine import ml_engine
from db.models import KnowledgeBase
from schemas import VerifyResponse, VerifyRequest, FactResult, MatchDetail
from db.connection import DB

router = APIRouter(prefix='/verify', tags=['Verification'])


def split_into_atomic_claims(complex_claim):
    complex_claim = complex_claim.strip().rstrip('.')
    split_pattern = r'(?:,\s*где\s+)|(?:,\s*а\s+)|(?:,\s*но\s+)|(?:,\s*котор[а-я]{2}\s+)|(?:,\s*и\s+)'
    parts = re.split(split_pattern, complex_claim)
    return [p.strip() for p in parts if len(p.strip().split()) >= 3] or [complex_claim]


def run_nli_model(contexts: list[str], claims: list[str]):
    inputs = ml_engine.tokenizer(
        contexts, claims, truncation=True, padding=True,
        max_length=512, return_tensors="pt"
    )
    inputs = {k: v.to(ml_engine.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = ml_engine.nli_model(**inputs)
        return torch.softmax(outputs.logits, dim=1).tolist()


@router.post('/', response_model=VerifyResponse)
async def verify_text(request: VerifyRequest, db: DB):
    atomic_claims = split_into_atomic_claims(request.text)
    all_results = []
    has_contradiction = False

    for claim in atomic_claims:
        query_vec_list = await run_in_threadpool(
            ml_engine.search_model.encode, [claim], normalize_embeddings=True
        )
        query_vec = query_vec_list[0]

        sim_expr = (1 - KnowledgeBase.embedding.cosine_distance(query_vec)).label('similarity')
        stmt = select(KnowledgeBase, sim_expr).order_by(
            KnowledgeBase.embedding.cosine_distance(query_vec)
        ).limit(3)

        db_result = await db.execute(stmt)
        matches_with_scores = db_result.all()

        if not matches_with_scores:
            all_results.append(FactResult(fact=claim, verdict="Нет данных", max_sim=0.0))
            continue

        max_sim = matches_with_scores[0].similarity

        if max_sim < request.threshold:
            all_results.append(FactResult(fact=claim, verdict="Нет данных", max_sim=max_sim))
            continue

        valid_candidates = []
        contexts_text = []
        for match, sim_score in matches_with_scores:
            if sim_score >= (request.threshold - 0.05):
                natural_lang = match.content.get("natural_language", '')
                clean_context = f"{match.entity_id} — это {natural_lang}"

                contexts_text.append(clean_context)
                valid_candidates.append((match, sim_score, clean_context))

        if not contexts_text:
            all_results.append(FactResult(fact=claim, verdict="Нет данных", max_sim=max_sim))
            continue

        probs_batch = await run_in_threadpool(run_nli_model, contexts_text, [claim] * len(contexts_text))

        current_fact_matches = []
        claim_lower = claim.lower()

        for i, probs in enumerate(probs_batch):
            match, sim_score, clean_ctx = valid_candidates[i]
            term_lower = match.entity_id.lower()

            subject_found = term_lower in claim_lower or any(w in claim_lower for w in term_lower.split() if len(w) > 3)

            n_val, e_val, c_val = 0.0, 0.0, 0.0
            for j, prob in enumerate(probs):
                label = ml_engine.id2label[j].lower()
                if 'entail' in label:
                    # Блокировка подмены субъекта
                    e_val = prob if subject_found else 0.0
                elif 'contradict' in label:
                    # Искусственный риск при несовпадении субъекта
                    c_val = prob if subject_found else 0.95
                else:
                    n_val = prob

            # ПЕССИМИСТИЧНЫЙ ФИЛЬТР (Neutral + Contradict > Entail)
            if (n_val + c_val) > e_val:
                e_val = 0.0

            res_scores = {"Подтверждено": e_val, "Нейтрально": n_val, "Противоречие": c_val}

            current_fact_matches.append(MatchDetail(
                term=match.entity_id,
                context=clean_ctx,
                similarity=sim_score,
                scores=res_scores,
                subject_match=subject_found
            ))

        current_fact_matches.sort(key=lambda x: x.scores.get('Подтверждено', 0), reverse=True)

        best_match = current_fact_matches[0]
        fact_verdict = max(best_match.scores, key=best_match.scores.get)

        if best_match.scores['Подтверждено'] == 0.0:
            temp = best_match.scores.copy()
            del temp['Подтверждено']
            fact_verdict = max(temp, key=temp.get)

        if fact_verdict == "Противоречие":
            has_contradiction = True

        all_results.append(
            FactResult(
                fact=claim,
                verdict=fact_verdict,
                max_sim=max_sim,
                best_match=best_match,
                all_matches=current_fact_matches
            )
        )

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