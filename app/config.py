import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DOCUMENTS_DIR = BASE_DIR / "documents" / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
MODELS_DIR = BASE_DIR / "models"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b"
)

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "350"))

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

TOP_K = int(os.getenv("TOP_K", "5"))

VECTOR_FILE = VECTOR_STORE_DIR / "vectors.npy"

METADATA_FILE = VECTOR_STORE_DIR / "chunks.json"