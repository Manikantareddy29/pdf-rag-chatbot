from fastapi import FastAPI

from app.routes import documents, chat

from app.services.vector_service import vector_store


app = FastAPI(
    title="PDF RAG Chatbot API",
    description="Backend API for PDF-based Retrieval Augmented Generation",
    version="1.0.0"
)


app.include_router(
    documents.router,
    prefix="/api/documents",
    tags=["Documents"]
)

app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["Chat"]
)


@app.get("/")
def home():

    return {
        "message": "PDF RAG Chatbot Backend is running",
        "status": "success"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "application": "PDF RAG Chatbot",
        "indexed_chunks": vector_store.count()
    }