from pydantic import BaseModel
from typing import List


class ChatRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    document: str
    page: int | str
    distance: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]