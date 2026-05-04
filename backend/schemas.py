from typing import Optional, List, Dict, Literal

from pydantic import BaseModel


class VerifyRequest(BaseModel):
    text: str
    threshold: float = 0.85

class MatchDetail(BaseModel):
    term: str
    context: str
    similarity: float
    scores: Dict[str, float]
    subject_match: bool


class FactResult(BaseModel):
    fact: str
    verdict: str
    max_sim: float
    best_match: Optional[MatchDetail] = None
    all_matches: List[MatchDetail] = []


class VerifyResponse(BaseModel):
    status: str
    has_contradiction: bool
    results: List[FactResult]





