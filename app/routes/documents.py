from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import DOCUMENTS_DIR

from app.services.rag_service import ingest_pdf, ingest_folder

from app.services.vector_service import vector_store


router = APIRouter()


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    filename = Path(file.filename).name

    destination = DOCUMENTS_DIR / filename

    try:

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty."
            )

        with open(destination, "wb") as output_file:
            output_file.write(content)

        result = ingest_pdf(destination)

        return {
            "message": "PDF upload and processing completed.",
            "result": result
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/ingest-folder")
def index_upload_folder():

    try:

        result = ingest_folder(
            DOCUMENTS_DIR
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("")
def get_documents():

    return {
        "documents": vector_store.list_documents(),
        "total_chunks": vector_store.count()
    }


@router.delete("/{filename}")
def delete_document(filename: str):

    safe_filename = Path(filename).name

    if safe_filename != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    removed_chunks = vector_store.remove_document(
        safe_filename
    )

    pdf_path = DOCUMENTS_DIR / safe_filename

    if pdf_path.exists():
        pdf_path.unlink()

    return {
        "message": "Document deletion completed.",
        "filename": safe_filename,
        "removed_chunks": removed_chunks
    }