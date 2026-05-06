import uuid
from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select, func

from db.connection import DB
from db.models import KnowledgeBase
from schemas import AddFactRequest, DeleteFactRequest, KnowledgeFactResponse
from core.ml_engine import ml_engine

router = APIRouter(prefix='/knowledge_base', tags=['Knowledge Base'])


@router.post('/', status_code=201)
async def add_fact(request: AddFactRequest, db: DB):
    stmt = select(KnowledgeBase).where(KnowledgeBase.entity_id == request.entity_id)
    result = await db.execute(stmt)
    existing_fact = result.scalars().first()

    if existing_fact:
        raise HTTPException(
            status_code=400,
            detail=f"Термин '{request.entity_id}' уже существует в базе данных."
        )

    embed_parts = [f"Термин: {request.entity_id}. Определение {request.entity_id}: {request.natural_language}"]

    if request.explanation:
        embed_parts.append(f"Иными словами: {request.explanation}")
    if request.examples:
        embed_parts.append(f"Примеры {request.entity_id}: {'; '.join(request.examples)}")
    if request.conflicts_with:
        embed_parts.append(f"Это противоречит: {'; '.join(request.conflicts_with)}")

    embed_text = " ".join(embed_parts)

    rich_context = (f"Термин: {request.entity_id} (Область: {request.domain}). "
                    f"Определение: {request.natural_language} ")
    if request.formal_logic:
        rich_context += f"Формально: {request.formal_logic}. "
    if request.explanation:
        rich_context += f"Пояснение: {request.explanation}. "
    if request.conflicts_with:
        rich_context += f"Логические анти-факты: {'; '.join(request.conflicts_with)}."

    embedding = await run_in_threadpool(
        ml_engine.search_model.encode, [embed_text], normalize_embeddings=True
    )

    new_fact = KnowledgeBase(
        id=str(uuid.uuid4()),
        entity_id=request.entity_id,
        domain=request.domain,
        fact_type=request.fact_type,
        content={
            "natural_language": request.natural_language,
            "formal_logic": request.formal_logic,
            "explanation": request.explanation,
            "examples": request.examples
        },
        relations={
            "parent_id": request.parent_id,
            "depends_on": request.depends_on,
            "conflicts_with": request.conflicts_with
        },
        embedding=embedding[0],
        search_text_bm25=embed_text,
        rich_context=rich_context
    )

    db.add(new_fact)
    await db.commit()

    return {
        "status": "success",
        "message": f"Термин '{request.entity_id}' успешно добавлен.",
    }


@router.delete('/', status_code=204)
async def delete_fact(request: DeleteFactRequest, db: DB):
    request.entity_id = request.entity_id.strip().lower()
    query = select(KnowledgeBase).where(func.lower(KnowledgeBase.entity_id) == request.entity_id)
    result = await db.execute(query)
    result = result.scalars().first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Факт {request.entity_id} не найден."
        )

    await db.delete(result)
    await db.commit()

    return None


@router.get('/{entity_id}', response_model=KnowledgeFactResponse)
async def get_fact(entity_id: str, db: DB):
    entity_id = entity_id.strip().lower()
    stmt = select(KnowledgeBase).where(func.lower(KnowledgeBase.entity_id) == entity_id)
    result = await db.execute(stmt)
    fact = result.scalars().first()

    if not fact:
        raise HTTPException(
            status_code=404,
            detail=f"Факт '{entity_id}' не найден."
        )

    return fact
