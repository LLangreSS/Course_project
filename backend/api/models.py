from pydantic import BaseModel
from typing import List, Optional

class VerifyRequest(BaseModel):
    text: str

class ClaimResult(BaseModel):
    claim: str
    verdict: str
    confidence: float
    source_text: str
    entity_name: Optional[str] = None

class VerifyResponse(BaseModel):
    results: List[ClaimResult]