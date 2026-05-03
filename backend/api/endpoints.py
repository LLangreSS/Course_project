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
        stmt = select(KnowledgeBase).order_by(
            KnowledgeBase.embedding.max_inner_product(query_vec)
        ).limit(3)

        db_result = await db.execute(stmt)
        matches = db_result.scalars().all()

        if not matches:
            all_results.append(FactResult(fact=claim, verdict="No data", max_sim=0.0))
            continue

        contexts_text = []
        for match in matches:
            natural_lang = match.content.get("natural_language", '')
            contexts_text.append(f"{match.entity_id} — это {natural_lang}")

        probs_batch = await run_in_threadpool(run_nli_model, contexts_text, [claim] * len(matches))
        best_match_detail = None
        highest_confirm = -1.0
        fact_verdict = "Нет данных"
        claim_lower = claim.lower()

        for i, probs in enumerate(probs_batch):
            match = matches[i]
            term_lower = match.entity_id.lower()
            subject_found = term_lower in claim_lower or any(w in claim_lower for w in term_lower.split() if len(w) > 3)

            n_val, e_val, c_val = 0.0, 0.0, 0.0

            for j, prob in enumerate(probs):
                label = ml_engine.id2label[j].lower()
                if 'entail' in label:
                    e_val = prob if subject_found else 0.0
                elif 'contradict' in label:
                    c_val = prob if subject_found else 0.95
                else:
                    n_val = prob

            if (n_val + c_val) > e_val:
                e_val = 0.0

            scores = {"Подтверждено": e_val, "Нейтрально": n_val, "Противоречие": c_val}

            if e_val > highest_confirm:
                highest_confirm = e_val
                best_match_detail = MatchDetail(
                    term=match.entity_id,
                    context=contexts_text[i],
                    similarity=1.0,
                    scores=scores,
                    subject_match=subject_found
                )

        if best_match_detail:
            fact_verdict = max(best_match_detail.scores, key=best_match_detail.scores.get)
            if best_match_detail.scores['Подтверждено'] == 0.0:
                temp = best_match_detail.scores.copy()
                del temp['Подтверждено']
                fact_verdict = max(temp, key=temp.get)

            if fact_verdict == "Противоречие":
                has_contradiction = True

            # 7. Записываем историю в БД для аналитики (асинхронно)
            #history_entry = VerificationHistory(
             #   claim=claim,
              #  verdict=fact_verdict,
               # confidence=best_match_detail.scores[fact_verdict],
                #matched_knowledge_id=matches[0].id  # Сохраняем связь с базой
            #)
            #db.add(history_entry)

        all_results.append(
            FactResult(
                fact=claim,
                verdict=fact_verdict,
                max_sim=1.0,
                best_match=best_match_detail
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