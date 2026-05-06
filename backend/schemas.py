from typing import Optional, List, Dict, Literal, Any

from pydantic import BaseModel


class VerifyRequest(BaseModel):
    text: str
    threshold: float = 0.85


class MatchDetail(BaseModel):
    term: str
    context: str
    similarity: float
    scores: Dict[str, float]


class FactResult(BaseModel):
    fact: str
    verdict: str
    best_match: Optional[MatchDetail] = None


class VerifyResponse(BaseModel):
    status: str
    results: List[FactResult]


class AddFactRequest(BaseModel):
    entity_id: str
    domain: str
    natural_language: str

    fact_type: Literal["definition", "theorem", "constraint"]

    explanation: Optional[str] = None
    formal_logic: Optional[str] = ""
    examples: List[str] = []

    parent_id: Optional[str] = ""
    depends_on: List[str] = []
    conflicts_with: List[str] = []


class DeleteFactRequest(BaseModel):
    entity_id: str


class KnowledgeFactResponse(BaseModel):
    entity_id: str
    domain: Optional[str]
    fact_type: Optional[str]
    content: Dict[str, Any]
    relations: Optional[Dict[str, Any]]





