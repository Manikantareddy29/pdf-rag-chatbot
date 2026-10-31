from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import TOP_K

from app.services.rag_service import answer_question


router = APIRouter()


class ChatRequest(BaseModel):

    question: str

    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=10
    )


@router.post("")
def chat(request: ChatRequest):

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        result = answer_question(
            request.question,
            request.top_k
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )