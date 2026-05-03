from typing import Optional, List

from pydantic import BaseModel


class VerifyRequest(BaseModel):
    text: str
    threshold: float = 0.85
class MatchDetail(BaseModel):
    term: str
    context: str
    similarity: float
    scores: dict
    subject_match: bool

class FactResult(BaseModel):
    fact: str
    verdict: str
    max_sim: float
    best_match: Optional[MatchDetail] = None

class VerifyResponse(BaseModel):
    status: str
    has_contradiction: bool
    results: List[FactResult]





