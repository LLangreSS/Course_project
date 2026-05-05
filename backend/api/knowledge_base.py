import uuid

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select

from db.connection import DB
from db.models import KnowledgeBase
from schemas import AddFactRequest, DeleteFactRequest
from core.ml_engine import ml_engine
from fastapi import status

router = APIRouter(prefix='/knowledge_base', tags=['Knnowledge Base'])

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

    text_to_embed = f"{request.entity_id}: {request.natural_language}"
    embedding = await run_in_threadpool(
        ml_engine.search_model.encode, [text_to_embed], normalize_embeddings=True
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
        embedding=embedding[0]
    )

    db.add(new_fact)
    return {
        "status": "success",
        "message": f"Термин '{request.entity_id}' успешно добавлен.",
    }


@router.delete('/', status_code=204)
async def delete_fact(request: DeleteFactRequest, db: DB):
    query = select(KnowledgeBase).where(KnowledgeBase.entity_id == request.entity_id)
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
