from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest
from app.services.rag_service import RAGService

router = APIRouter(
    prefix="/chat",
    tags=["Financial RAG"]
)

rag = RAGService()


@router.post("/")
def chat(request: ChatRequest):

    try:

        response = rag.ask(request.question)

        return response

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )